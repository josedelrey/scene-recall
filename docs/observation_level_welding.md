# Experimental observation-level RGB-D loop welding

SceneRecall has an opt-in observation-level welding path. It retains native RGB-D
keyframe factors independently of the local smoother's landmark budget, joins
verified historical identities, jointly refines covisible cameras and landmarks,
and publishes pose revisions through the existing versioned reconstruction path.
It remains experimental. The sparse online defaults are unchanged.

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
Linux suite passes 1,025 tests. A fresh RPY 120 replay after the final state
publication cleanup matches saved trajectory, atlas, and reconstruction arrays
exactly. Windows and hosted CI were not exercised.

## Reproduction and remaining work

The ignored `outputs/observation_welding/` directory contains the selection
protocol, raw policy runs, causal and revised trajectories, registration events,
per-frame solver diagnostics, exact reconstruction audits, and fixed-camera map
scores. Each run stores a protocol and source/package digests. Reproduce a pair
with fresh output paths using the same raw source and options, adding
`--observation-welding` only to the second policy. For ScanNet add
`--local-exposure static` to both policies. Score saved maps with
`experiments/evaluate_online_maps.py` under a common run directory.

The current result is a bounded welding layer rather than a single global
maximum-posterior estimator. The pose graph still uses correlated odometry and
registration summaries for topology and initial placement. Cameras outside a
weld are conditioned through its boundary, and historical marginal priors do
not enter the global joint objective. Global landmark revisions do not yet feed
back into local tracking. Nearest-camera correction can introduce piecewise
deformation outside the optimized neighborhood. Intrinsics, color/depth
extrinsics, exposure timing, and coherent depth bias remain fixed or assumed.
The graph's 512-node limit bounds the ledger, but there is no eviction and
replayable prior scheme for longer sessions.

The next step is an incremental shared camera-landmark graph that carries
cross-boundary square-root priors or replays their raw observations when loop
corrections invalidate a linearization. It must replace overlapping pose
summaries in the optimization objective, revise the active local map, and
validate new sensors and captures before promotion. Preserve the raw ownership,
reversible association, causal publication, and exact refusion contracts from
this experiment.
