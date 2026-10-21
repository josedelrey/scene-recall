# Verified loop graphs and place retrieval

The experimental global graph improves spatial accuracy substantially on full
XYZ, Desk and RPY, while keeping the original local trajectories and tracking
failures intact. Generalization and reconstruction remain mixed. Sparse, the
local SLAM estimator and the original mean-signature forest remain unchanged
defaults. This investigation does not promote a production recovery policy.

## Research and design choice

The [recovery atlas](slam_recovery.md) verifies historical camera registrations
and connects local gauges with a forest. It cannot revise drift around a loop.
Earlier RPY audits showed both accumulated drift and biased registrations, while
Pioneer exposed retrieval failures and scarce metric matches. These findings
motivate testing recognition, confirmation and global correction separately.

The following primary work informed the investigation. Published results are
architectural evidence, not matched SceneRecall competitor measurements.

| Approach | Relevant mechanism | Decision for this experiment |
| --- | --- | --- |
| [ORB-SLAM3](https://arxiv.org/abs/2007.11898) | Atlas retrieval, geometric verification in covisible views, map merging and loop correction | Retain lost-map identities and require different source and target observations before adding a relationship |
| [AnyLoc](https://arxiv.org/abs/2308.00688) and [SALAD](https://arxiv.org/abs/2311.15937) | Rich local representations and feature aggregation improve place proposals | Replace the mean descriptor with direct local descriptor voting as an auditable CPU control, while keeping geometric acceptance separate |
| [Graduated non-convexity](https://arxiv.org/abs/1909.08605) | Continuation and an explicit outlier process reduce sensitivity to false constraints | Implement a graduated Geman-McClure loop objective and compare it with linear least squares on identical factors |
| [PCM](https://robots.engin.umich.edu/publications/jmangelson-2018a.pdf) | Pairwise consistency checks help select inter-map measurements | Confirm compatible hypotheses before graph admission and abstain on competing disconnected explanations. No maximum-clique PCM solver is implemented |
| [DROID-SLAM](https://arxiv.org/abs/2108.10869) and [MASt3R-SLAM](https://arxiv.org/abs/2412.12392) | Learned correspondence or two-view geometry coupled with revisable optimization | Preserve a measurement boundary for a future learned frontend. Native metric depth remains the only depth origin in this experiment |
| [Kimera2](https://arxiv.org/abs/2401.06323) | Robust metric-semantic estimation and pose/mesh graph correction | Keep observations, local identities and revised global placements separate for future persistent semantic memory |

No external implementation, learned weights or new dependencies are imported.
The available NVIDIA driver did not initialize on the research host. Learned
retrieval remains a useful next controlled comparison, particularly if it can
reduce the expense of the direct descriptor-voting control. Learned pointmaps
would need separate provenance and uncertainty, including checks against native
depth and metric scale. They must not become additional independent depth factors
by assumption.

## Implemented architecture

`slam/retrieval.py` counts mutual two-neighbor descriptor matches at ratio 0.75.
Usable source-depth matches rank first, total image matches break ties, then the
original mean-signature distance and frame index break remaining ties. Missing
target depth does not remove image support. Votes propose candidates only.
`AtlasConfig(retrieval="descriptor_votes")` enables this optional forest
retriever. The default is still `"mean"`.

Both retrievers retain the same immutable snapshots and the same bounded bank:
64 entries, 16 recent entries, cadence five and deterministic historical
retention. Both verify up to four candidates per query with the unchanged historical
registration gates. There is no relaxed geometric threshold or reference-based
candidate selection.

The separate graph experiment also searches within the current local segment,
with a minimum gap of 30 inputs. Cross-segment candidates retain a ten-input gap.
The mean and vote graph variants each verify four candidates from this expanded
candidate set. Forest-versus-graph comparisons therefore include the addition
of within-segment constraints. Mean-versus-vote graph comparisons isolate the
retrieval rule on the same bank and candidate budget.

`slam/loop_confirmation.py` requires two compatible registrations with different
source frames and different target frames. Hypotheses compare their implied
local-gauge relationships within 50 mm and 3 degrees. Thirty-input source and
target cells define evidence groups. Each group needs its own confirmation.
A query with verified candidates from multiple disconnected source components
abstains. Confirmation is causal, although the final graph solve is offline.
These checks cannot identify an exact geometric place alias.

`slam/pose_graph.py` supplies a bounded dense reference solver:

- Nodes retain immutable frame identities, local segment identities and supplied
  local poses. Consecutive selected cameras within one segment receive odometry
  summaries. No odometry edge crosses a local tracking failure.
- Raw loop measurements remain camera-to-camera rigid transforms. The residual
  chart is translation and rotation of
  `inverse(Z_target_source) @ inverse(X_target) @ X_source`.
- Engineering scales are 10 mm and 1 degree for local summaries, and 30 mm and
  2 degrees for loops. Each evidence group's total loop information is divided
  by its factor count to limit repeated-view information multiplication.
- Loop cost is Geman-McClure, with normalized innovation `s`, cost
  `c² s / (c² + s)` and weight `(1 + s / c²)^-2`. The continuation uses cutoffs
  24, 12, 6 and 3. Local summaries remain linear. This local solver is not a
  certified global optimizer and does not reproduce the paper's non-minimal
  solver machinery or outlier guarantees.
- Weights below 0.1 exclude loops. The graph rebuilds gauges and refits until
  exclusion stabilizes. Rejected measurements remain available for audit.
  Each disconnected component fixes its earliest camera at its supplied local
  pose. A discarded bridge cannot leave a stale component transform.
- `revoke` removes an explicit loop. `revise` refreshes local summaries while
  retaining raw camera-to-camera loop measurements. Call `solve` after these
  mutations to recompute placements, components, affected keyframes and version.
  Mutation does not itself provide a new solved placement.

The graph admits at most 512 nodes. It uses analytic edge Jacobians, damped
normal solves and a finite iteration budget. Group budgets are a correlation
heuristic, not calibrated covariance. Correlations between groups and between
local summaries and loop measurements remain unmodeled.

`experiments/slam_loop_graph.py` interpolates left pose corrections only between
keyframes of the same local segment. It writes separate `atlas.npz` exports with
component gauges, original source-frame lineage, local segment IDs and a version.
The standard local `trajectory.npz` is copied byte for byte. Original statuses,
failures and adjacent transitions remain unchanged.

Reconstruction rebuilds admitted raw depth using the final corrected cameras.
It does not transform a previously fused surface as if all local errors were
rigid. There is no live TSDF reintegration or frontend feedback. Future semantic
observations should attach to original frame evidence and follow revised camera
placement, while retaining revocable place identities.

## Frozen evaluation

The study freezes 13 natural populations from eight independently recorded
captures, covering 4,752 distinct inputs across TUM, ScanNet and ETH3D. Pioneer
windows overlap its full recording. Two windows from one capture are not
independent scene replicates. Five controlled RGB outages supplement these
populations. Short XYZ and the previously identified Pioneer retrieval probe are
development cases. Existing recordings are research evidence, not hidden tests.

Six policies replay the same local trajectories: original forest, vote forest,
mean graph with linear and robust losses, and vote graph with linear and robust
losses. Both forest policies keep the same bank admission. The original forest
reproduces the saved scientific pose and component arrays exactly in all 18
cases. All six methods preserve the same initial-root population in every case.
Consequently, the gains below cannot come from adding easier root frames.

References never enter retrieval, registration, confirmation, graph factors or
optimization. They score results after those results are frozen. Errors below
are translation and rotation RMSE after fixing the first referenced pose of each
component. They are **root-anchored errors**, not position-aligned ATE.

| Natural population | Root frames | Forest translation, mm | Mean robust graph, mm | Vote robust graph, mm | Forest / vote graph rotation, degrees |
| --- | ---: | ---: | ---: | ---: | ---: |
| XYZ full | 792 | 80.84 | 23.46 | 26.61 | 1.46 / 1.09 |
| Desk full | 573 | 116.57 | 72.30 | 63.67 | 5.65 / 3.26 |
| RPY full | 694 | 118.69 | 67.68 | 62.24 | 7.96 / 4.62 |
| Walking early | 300 | 72.23 | 54.26 | 53.55 | 1.47 / 0.95 |
| Walking late | 300 | 139.07 | 140.13 | 141.19 | 2.50 / 2.79 |
| Repetitive early | 300 | 9.75 | 9.35 | 8.79 | 0.83 / 0.89 |
| ScanNet 1000 | 200 | 80.26 | 80.26 | 80.26 | 1.99 / 1.99 |
| ScanNet 3500 | 200 | 103.71 | 102.83 | 101.62 | 1.85 / 1.86 |
| Pioneer early | 69 | 22.01 | 22.89 | 22.89 | 0.30 / 0.29 |
| Pioneer 400 | 70 | 163.87 | 163.87 | 163.87 | 3.94 / 3.94 |
| Pioneer full | 69 | 22.01 | 22.89 | 22.89 | 0.30 / 0.29 |

Late repetitive has a singleton initial root, so its zero anchored error provides
no useful global accuracy evidence. The lab root is also a singleton and lacks
a valid reference. Neither case recovers its initial map. Across the eleven
natural populations with at least three root references, vote robust translation
improves six, worsens three and is unchanged on two. Overlapping Pioneer entries
must not be counted as independent corroboration.

The graph does not solve Pioneer continuity. Full Pioneer retains 253 local
failures and only 69 initial-root frames. Voting increases confirmed graph loops
from 13 to 17, but does not produce another root connection. Metric support,
confirmation and disconnected ambiguity remain bottlenecks after retrieval.

Controlled outages retain their original losses. XYZ80, Desk, Walking and early
repetitive recover respectively 70, 563, 290 and 290 root frames under every
policy. Vote robust translation RMSE changes from 18.35 to 18.00 mm, 128.49 to
58.69 mm, 63.04 to 48.77 mm, and 11.74 to 9.62 mm. XYZ80 rotation slightly worsens.
Late repetitive remains unrecovered. Its blanking interval overlaps an already
fragile regime and does not add a new root recovery success.

## External geometry and false constraints

The robust loss is useful insurance against some contradictory factors, rather
than a general accuracy improvement. On clean RPY, the linear vote graph reaches
60.36 mm and 4.52 degrees, better than robust's 62.24 mm and 4.62 degrees.
Walking early similarly favors linear translation, 47.16 versus 53.55 mm.
The mean graph is better than vote retrieval on full XYZ.

Seven of 224 reference-scored RPY vote-graph loop factors exceed 10 cm or
5 degrees. All survive the robust policy. Their weights range approximately
0.49–0.94. Some are coherent with the biased local trajectory, so small internal
innovation is not an external correctness certificate. Another four active RPY
loops lack reference endpoints and cannot be scored. Tight 20 mm / 1 degree
accuracy is missed by 172 of the 224 scored factors. These are reference
inconsistencies, not correspondence-level physical place labels.

The mean robust graph rejects one late-Walking constraint with 132.8 mm reference
translation error. The vote robust graph rejects no loops in the natural matrix,
even though some externally inaccurate factors remain. A rejection count alone
therefore does not measure false-association robustness.

The saved synthetic controls isolate backend behavior:

| Control | Linear translation RMSE | Robust translation RMSE | Interpretation |
| --- | ---: | ---: | --- |
| Clean biased local chain with correct loops | 24.97 mm | 26.63 mm | Robustness can cost clean-data accuracy |
| Two injected false cycles | 2,288.82 mm | 26.63 mm | Both false cycles are excluded by robust fitting |
| Coherent 10% metric scale bias | 170.22 mm | 170.22 mm | Near-zero cost cannot reveal shared metric bias |
| False bridge without cycle support | 3,835.96 mm | 3,835.96 mm | Both methods confidently join the wrong gauge |

Disconnected synthetic maps retain separate gauges until a measurement connects
them. Unit tests also demonstrate that a correct loop can be rejected when
local drift is large. Exact place aliases, correlated registration error and
incorrect bridges need independent evidence, not a stronger robust-loss claim.

## Downstream reconstruction

The primary cohort freezes the first 24 cadence-ten training candidates and
interleaved targets before scores. Fusion withholds target depth, while tracking
and registration may use those target images and depths. This is reconstruction
validation, not an independent estimator measurement holdout or novel-view test.
All methods render from the same reference target cameras. Missing training
references withhold a case without replacement.

The score averages absolute target-depth error capped at 200 mm, assigning that
same penalty to missing predictions. It therefore includes coverage loss. Surface
and TSDF mesh results remain separate.

| Primary cohort | Forest surface / graph surface, mm | Forest mesh / graph mesh, mm | Forest / graph mesh coverage |
| --- | ---: | ---: | ---: |
| XYZ | 43.72 / 46.44 | 33.66 / 26.97 | 94.71% / 95.31% |
| Desk | 66.76 / 69.78 | 52.29 / 51.09 | 90.90% / 91.87% |
| RPY | 75.75 / 73.43 | 71.76 / 62.30 | 87.52% / 89.02% |
| Early repetitive outage | 22.36 / 21.99 | 27.81 / 26.01 | 91.77% / 92.10% |
| Late repetitive outage | 171.98 / 171.98 | 175.02 / 175.02 | 16.31% / 16.31% |

The graph columns use vote robust. Lab and Walking outage comparisons are
withheld for missing fixed training references. RPY uses 24 training views under
both forest and graph, so the mesh improvement is a geometry correction. Its
original unconnected local map uses only 12 views, with mesh error 98.78 mm and
coverage 68.34%. Early repetitive's large gain from the local four-view map to
the connected 23-view map already belongs to historical recovery. The graph adds
a smaller refinement to that recovered geometry.

A separate **post hoc** full-extent XYZ comparison uses 80 training candidates
and 79 referenced targets. Mesh error falls from 72.45 to 23.16 mm, with coverage
97.02% to 98.16%. Surface error is essentially unchanged, 42.51 to 42.56 mm.
Reference-pose mesh fusion scores 19.36 mm and is a diagnostic control, not
independent physical surface truth. This full-extent result is kept separate
from the primary bounded-map ranking.

Full-extent RPY exceeds the unchanged TSDF scratch-grid capacity. The failed run
and its protocol are retained, without a TSDF score or a silently enlarged
budget. A surface-only continuation uses the same full-extent cohort and lowers
surface error from 79.08 to 74.33 mm. This does not supply a missing mesh result.

## Reproduction and artifact contracts

Start from an unrevised `control` run produced by the
[recovery workflow](slam_recovery.md). Its source RGB-D recording must be
available for raw replay. Choose fresh output directories.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python \
  experiments/slam_loop_graph.py \
  --local-run outputs/recovery/control \
  --output-dir outputs/loop_graph/raw

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python \
  experiments/slam_loop_graph.py \
  --local-run outputs/recovery/control \
  --features outputs/loop_graph/raw/query_features.npz \
  --output-dir outputs/loop_graph/cached

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python \
  experiments/evaluate_loop_graph_maps.py \
  --run outputs/loop_graph/raw \
  --output-dir outputs/loop_graph/maps

uv run --locked python experiments/slam_graph_controls.py \
  --output-dir outputs/loop_graph/controls
```

Raw extraction verifies the entire canonical source identity and extracts query
observations. A supplied feature cache must match its extraction hash and input
cohort or the exact local-report hash. SHA256 provenance records the source,
local archive, configurations, package and driver. The driver shares identical
registration and voting computations between variants to reduce matrix work.
Its per-policy matrix timings are consequently shared profiles, not comparable
end-to-end speed measurements.

`run_loop_graph_study.py` accepts a JSON manifest mapping case names to
`local_run` and optional `features` paths, with explicit logs and up to four
workers. `summarize_loop_graph_study.py` reports fixed populations, separate
component gauges, active-loop external errors and reconstruction. Concurrent
matrix timings must not be used as isolated performance claims.

The ignored study root is `outputs/slam_graph_study/`. It contains the frozen
protocol, case manifest, original-file protection hashes, final matrix, raw
constraints and accepted proposals, synthetic controls, reconstruction products,
source snapshots, repeat checks and isolated timing. Older development outputs
are labeled separately and are excluded from primary conclusions.

## Performance and verification

Isolated opposite-order runs on a resident 200-input Pioneer prefix take
5.58–5.88 seconds for the original atlas and 52.36–52.37 seconds for descriptor
voting. Their atlas-only throughput is 34.0–35.9 versus 3.82 frames/second, with
per-input p95 latency 84–91 ms versus 973–1,011 ms. These figures exclude feature
extraction and local tracking. The exhaustive descriptor-voting control is too
expensive to justify promotion on the present continuity evidence.

The resident RPY packet has 141 nodes and 367 edges. Opposite-order linear solves
take 247–250 ms and robust solves 1.38–1.40 seconds. Build and dense export add
about 25 ms. All repeated graph arrays match the primary output exactly. Warm
inputs, short atlas timing, a dense batch solve and one machine limit these
measurements. They do not establish end-to-end online performance. Peak resident
memory for the combined timing process is approximately 200 MiB. Primary RPY
reconstruction reaches approximately 748 MiB, and its full-extent TSDF still
exceeds the configured grid budget.

All 4,591 protected prior scientific files remain byte-identical. All 108 local
policy archives preserve their source trajectory bytes, frame diagnostics,
statuses, evaluation, configuration and input identity. Each of the 18 original
forest comparisons reproduces its saved poses and component IDs exactly.
The final XYZ80 cached repeat and independent raw repeat each reproduce 41 atlas
and factor arrays plus event diagnostics exactly. The XYZ reconstruction repeat
reproduces 26 prediction/mesh arrays, fusion cohorts and scores exactly.

The Linux locked environment passes 963 tests, Ruff lint, formatting and
whitespace checks. No dependency or lockfile change is needed. Native Windows
and hosted CI were not run. No commit, push or context synchronization is made.

## Remaining limitations and next work

The strongest result is corrected long-horizon geometry on three familiar TUM
recordings. Continuity on strongly fragmented recordings remains unresolved.
No new hidden benchmark, learned frontend, IMU fusion, persistent cross-session
store, calibrated posterior or production semantic memory is claimed.

Prioritize depth-aware spatial support and pooled covisible anchors, preserving
measurement identity and correlated pose uncertainty. Test a learned retriever
against the direct-voting control with the same verification budget and explicit
negative places. Improve independent registration validation under rotation,
exposure motion, calibration error and native-depth dropout. Then replace the
dense graph with a sparse incremental backend and evaluate correction delivery
to a live reconstruction and semantic memory consumer. More constraints and
lower residuals remain insufficient promotion criteria.
