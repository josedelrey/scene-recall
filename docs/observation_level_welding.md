# Experimental observation-level RGB-D loop welding

SceneRecall has an opt-in observation-level welding path. It retains native RGB-D
keyframe factors independently of the local smoother's landmark budget, joins
verified historical identities, jointly refines covisible cameras and landmarks,
and publishes pose revisions through the existing versioned reconstruction path.
It remains experimental. The sparse online defaults are unchanged.

The subsequent [persistent shared estimation experiment](shared_estimation.md)
replaces the conditional welding overlay with one replayable raw camera–landmark
objective. This document retains the earlier method and its comparison evidence.

## Architecture

`--observation-welding` enables a separate raw-factor ledger in
`slam/system.py` and `slam/observation_welding.py`. Every admitted global
keyframe can retain up to 128 spatially distributed frontend observations. The
selection reserves capacity for previously seen identities and new identities
with native depth. Each entry records its feature row, frontend identity, native
depth sample index, RGB pixel, depth-camera point, exposure transform, and frozen
depth whitening. A native sample has at most one owner within its view. The
ledger retains observations even when the bounded local smoother cannot admit
another landmark. Segment and frontend identity form its persistent track key.

Historical retrieval, registration, distinct-view confirmation, and graph edge
revocation remain the admission gates. An admitted loop contributes reversible
track equivalences using only fitting correspondences that have valid depth in
both views and pass metric and image checks. The other descriptor rows provide
post-fit cross-view validation. Active loop edges determine identity unions on
each solve, so revoking an edge removes its identity claims without rewriting
native packets.

The validation rows are withheld from loop association construction, not from
all other estimation. Some may share frontend tracks with the joint objective.
They are a contradiction gate, not an independent accuracy measurement.

After the scheduled pose graph solve, the welding backend selects up to two
recent loops. For each it chooses up to eight cameras by shared track support,
then up to 96 landmarks observed by at least two cameras. The joint RGB-D
objective reuses the existing direct residuals and bounded camera-landmark
solver. One camera fixes the gauge. A candidate must preserve withheld loop
agreement before its camera and point state is committed. Previously solved
points remain in the ledger and are transported by their reference camera when
the graph placement changes. This avoids seeding every later solve from a
newly backprojected point.

The pose graph supplies component topology and initial placement. A loop pose
edge that its robust pose-only solve rejects may still support a same-component
observation weld after independent RGB-D verification. Cross-component welds
currently require an accepted graph bridge. Camera corrections propagate to
other graph nodes in each touched segment through the nearest selected camera.
`OnlineSLAM` publishes a new pose version and dirty frames. Reconstruction
rebuilds affected chunks from original camera-local samples. Explicit loop
revocation recomputes topology, live identity unions, placement, and fusion.

The local joint smoother and global welding cannot be enabled together. They
would reuse the same raw evidence through a local marginal prior and global
factors without carrying their cross-state correlation. `--local-exposure
static` permits a deliberate static color/depth assumption when a source has
no usable exposure clock. It is not the default.

## Active-state feedback follow-up

`--welding-feedback landmark`, `camera`, or `joint` enables a second opt-in
experiment alongside `--observation-welding`. Omitting the mode after the flag
selects `landmark`, the mode used for the transfer evaluation below. A solve can
transfer live, loop-supported point states, active keyframe camera states, or
both into the tracker. The transfer is a state
revision. It does not add the global estimate as another probabilistic factor.

For a selected RGB camera pose `C_i` and landmark `X_j`, welding minimizes
robust blocks of the original observations. Its image residual is
`project(C_i^-1 X_j) - pixel_ij`. Its native-depth residual is
`W_ij (E_ij^-1 C_i^-1 X_j - depth_ij)`, where `E_ij` is the recorded depth-to-RGB
exposure transform and `W_ij` is frozen depth whitening. The bounded window
fixes one camera gauge and conditions on cameras outside it. There is one raw
factor per admitted view and feature. Loop associations only change landmark
equivalence classes while their verified edges remain live.

For active segment `s`, let `G_a` be the globally revised depth pose of its
latest graph node and `L_a` its current tracker pose. The chart transform is
`A_s = G_a L_a^-1`. Camera feedback sets active tracker poses to
`L_i' = A_s^-1 G_i`. Landmark feedback sets `x_j' = A_s^-1 X_j`. When a new
graph node `q` follows node `p`, its odometry measurement uses only the new
tracker-relative motion, `Z_qp = L_q^-1 L_p`, and its graph-local chart extends
as `H_q = H_p Z_qp^-1`. Previously recorded odometry edges are never replaced
with the loop-corrected global displacement. Published nonkeyframes follow
their revised active anchors, and reconstruction still rebuilds from original
camera-local samples.

Cached point states record every supporting loop edge. Explicit revocation
removes its associations and dependent point states. The active tracker releases
such points to its latest retained native observation when available. Rejected
cross-component bridges cannot supply point feedback. The online local joint
optimizer remains disabled in feedback modes. Its inactive marginal prior is
dropped after a feedback revision because a nonlinear historical correction
cannot safely be inserted into that frozen chart. Retained local raw packets
remain, but consumed historical information is not replayed into a new local
prior. This preserves single ownership within each optimization objective at
the cost of lost local historical information and occasional local graph
restarts. Global raw packets remain available within the 512-node graph budget.

## Evaluation

The raw causal comparisons use CLAHE, half-resolution SIFT, flow recovery, and
the same mapping settings for control and welding. References are removed before
estimation. The first XYZ and RPY 120-frame cases were exploratory. A subsequent
protocol froze RPY 240, Desk 240, and ScanNet 160 before their transfer runs.
The RPY 400 extension followed inspection of RPY 240. These runs cover 920
distinct inputs across three TUM recordings and one previously studied ScanNet
capture. Prefixes and paired policies repeat observations. Desk is another
scene with the same TUM sensor. ScanNet supplies another sensor, with static
exposure assumed in both compared policies.

Values below are control / welding. Translation is root-connected RMSE in mm.
Surface error is the capped 20 cm fixed-reference-camera target score in mm.
These target depths are independent of fusion, although their frames may enter
tracking and estimation. Reference-pose fusion is not physical surface truth.

| Recording | Causal translation | Causal rotation, degrees | Final revised translation | Surface error | Welded solves |
| --- | ---: | ---: | ---: | ---: | ---: |
| TUM XYZ 0:120 | 14.54 / 16.53 | 0.625 / 0.570 | 15.04 / 16.36 | 52.76 / 54.22 | 17 |
| TUM RPY 0:120 | 109.85 / 86.15 | 5.214 / 4.777 | 109.85 / 66.32 | 88.19 / 82.12 | 1 |
| TUM RPY 0:240 | 119.61 / 89.01 | 5.383 / 5.046 | 109.14 / 107.71 | 88.85 / 90.57 | 12 |
| TUM RPY 0:400 | 122.56 / 104.51 | 5.394 / 5.130 | 113.65 / 118.07 | 88.59 / 87.52 | 34 |
| TUM Desk 0:240 | 48.62 / 48.62 | 3.063 / 3.063 | 48.63 / 48.63 | 57.75 / 57.75 | 0 |
| ScanNet 0:160, static exposure | 91.41 / 84.42 | 2.496 / 2.381 | 91.22 / 82.14 | 63.38 / 60.39 | 5 |

All six comparisons retain the same local transition failures and root
population between policies. Desk has no eligible welds in this window. With
the timed exposure policy, ScanNet has no usable native factor packets and
reproduces the pose-only baseline. The static-exposure result should therefore
be read as a separate model assumption.

XYZ demonstrates the main unresolved risk. Withheld loop errors improve sharply,
yet external translation, final orientation, surface error, and surface coverage
worsen. RPY improves early but its 240-frame surface score worsens, while its
400-frame final revised translation worsens. No internal cost or loop agreement
criterion reliably predicts all external outcomes. The method has useful gains
on RPY and this ScanNet capture, without evidence for default promotion or
unseen-scene generalization.

The complete 400-frame RPY runs process 5.20 / 4.72 inputs per second, with
process p95 of 0.571 / 0.601 seconds and peak RSS of 252.7 / 257.3 MiB.
XYZ 120 slows from 8.34 to 5.08 inputs per second. These are serial raw runs
on one host, not real-time guarantees. Across the scored runs, every saved
reconstruction chunk matches fresh native-sample fusion exactly. The full
Linux suite passes 1,029 tests after the feedback follow-up. A fresh RPY 120
replay after the initial state publication cleanup matches saved trajectory,
atlas, and reconstruction arrays exactly. Windows and hosted CI were not
exercised.

The feedback follow-up reused the frozen settings and raw controls above. XYZ
120 and RPY 120 were diagnostics. The RPY 240, Desk 240, and ScanNet 160 transfer
cases were named before feedback runs, followed by an RPY 400 extension. This
again covers 920 distinct inputs across four recordings, with repeated RPY
prefixes and policy ablations. The table gives control / welding / landmark
feedback. All translation values are root-connected RMSE in mm. Surface error
uses the same fixed-camera score as above.

| Recording | Causal translation | Final revised translation | Surface error |
| --- | ---: | ---: | ---: |
| TUM XYZ 0:120 | 14.54 / 16.53 / 16.26 | 15.04 / 16.36 / 15.58 | 52.76 / 54.22 / 57.05 |
| TUM RPY 0:240 | 119.61 / 89.01 / 86.94 | 109.14 / 107.71 / 106.42 | 88.85 / 90.57 / 89.59 |
| TUM RPY 0:400 | 122.56 / 104.51 / 104.55 | 113.65 / 118.07 / 119.17 | 88.59 / 87.52 / 90.20 |
| TUM Desk 0:240 | 48.62 / 48.62 / 48.62 | 48.63 / 48.63 / 48.63 | 57.75 / 57.75 / 57.75 |
| ScanNet 0:160, static exposure | 91.41 / 84.42 / 82.75 | 91.22 / 82.14 / 80.14 | 63.38 / 60.39 / 57.99 |

No transfer run acquired a new tracking loss. Median map inliers change only
slightly, such as 74 to 72 on RPY 400 and 67 to 68 on ScanNet 160 relative to
welding. The landmark-only path improves ScanNet trajectory and capped surface
error, although surface coverage drops from 0.8655 to 0.8614. XYZ remains
worse than control and its surface error worsens beyond welding alone. RPY 240
improves translation over welding, but RPY 400 reverses that gain in final
trajectory and map geometry. Causal rotation is also mixed. At RPY 240,
camera-only and joint feedback end at 121.74 and 120.78 mm revised translation,
with 91.42 and 91.46 mm surface error. These camera modes are retained for
diagnosis, not recommended use.

The 400-frame RPY landmark run processes 3.99 inputs per second versus 4.72
for welding alone, with process p95 0.806 versus 0.601 seconds and peak RSS
257.9 versus 257.3 MiB. Its 80 graph nodes do not exercise the 512-node limit.
All scored feedback reconstruction chunks match fresh native-sample fusion
exactly. An independent 60-frame XYZ replay matches every emitted trajectory
and atlas array in the 120-frame run over the common prefix. The extra local
prior resets increase local graph restarts from zero to two on XYZ 120 and from
three to four on RPY 240. These are internal maintenance failures, not lost
tracking transitions.

An unchanged-control XYZ 60 replay after the code changes also matches every
emitted trajectory and atlas array in its saved 120-frame control prefix.

On RPY 400, the final poses of the first 240 frames move a median 109 mm from
their emitted positions with landmark feedback, versus 33 mm with welding
alone. The feedback run admits two additional observation-welding events after
frame 240. A subsequent [matched-horizon audit](shared_estimation.md) found
that the feedback run already has a 109.66 mm median emitted-to-final shift at
240 inputs, versus 108.84 mm over that same prefix after 400 inputs. The
240-to-400 pose change has 8.64 mm RMSE and 18.71 mm p95. The additional events
therefore do not by themselves explain the large earlier shift. Event-level
causal attribution remains unresolved.

The feedback implementation proves a causal state transport and revocation
contract, but these results do not establish a general accuracy improvement.
The independent ScanNet result depends on an explicit static exposure
assumption, Desk has no effective weld, and the two TUM motion cases remain
mixed. Lower internal objective cost and correct reconstruction refusion are
insufficient evidence for default promotion.

## Reproduction and remaining work

The ignored `outputs/observation_welding/` directory contains the selection
protocol, raw policy runs, causal and revised trajectories, registration events,
per-frame solver diagnostics, exact reconstruction audits, and fixed-camera map
scores. Each run stores a protocol and source/package digests. Reproduce a pair
with fresh output paths using the same raw source and options, adding
`--observation-welding` only to the second policy. For ScanNet add
`--local-exposure static` to both policies. Score saved maps with
`experiments/evaluate_online_maps.py` under a common run directory.

The ignored `outputs/observation_feedback/` directory contains its frozen
selection protocol, raw feedback runs, paired links to preserved controls,
map audits, and the prefix replay. Add `--welding-feedback landmark` to the
welding command to reproduce the selected feedback policy. `camera` and `joint`
are diagnostic ablations. Run the fixed-camera evaluator with all three
policies under a common directory.

The current result is a bounded welding layer rather than a single global
maximum-posterior estimator. The pose graph still uses correlated odometry and
registration summaries for topology and initial placement. Cameras outside a
weld are conditioned through its boundary, and historical marginal priors do
not enter the global joint objective. Opt-in feedback changes tracker states
without a shared local-global factorization. Nearest-camera correction can
introduce piecewise deformation outside the optimized neighborhood. Intrinsics, color/depth
extrinsics, exposure timing, and coherent depth bias remain fixed or assumed.
The graph's 512-node limit bounds the ledger, but there is no eviction and
replayable prior scheme for longer sessions.

The next step is an incremental shared camera-landmark graph that carries
cross-boundary square-root priors or replays their raw observations when loop
corrections invalidate a linearization. It must replace overlapping pose
summaries in the optimization objective and replace feedback overlays with
shared active optimizer states. Validate new sensors and captures before
promotion. Preserve the raw ownership,
reversible association, causal publication, and exact refusion contracts from
this experiment. The [iSAM2 Bayes-tree formulation](https://www.cs.cmu.edu/~kaess/pub/Kaess12ijrr.pdf)
is a useful model for incremental relinearization, but this implementation
does not yet have its shared nonlinear factorization or marginal recovery.
