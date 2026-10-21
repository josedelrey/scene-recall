# Incremental SLAM landmark lifecycle investigation

Historical local outputs were [compacted into evidence archives](output_storage.md).
Restore the required reports and traces before using the saved-run examples below.

The [initial incremental SLAM study](incremental_slam.md) remains the baseline.
This follow-up investigates landmark capacity, rank-deficient retirement and
measurement persistence. Sparse remains the production default. The original
eight-camera, 128-point SLAM configuration also remains unchanged by default.
The new policies are experimental options, without a general accuracy claim.

## Diagnosis and architectural decision

The baseline's fixed point budget can become occupied by landmarks outside the
current view. On full Desk, the baseline accepts 13 joint updates, rejects 66
windows for multiview support and restarts its graph six times. Some admitted
cameras have only two observations of existing graph points. Subsequent camera
and point retirement can therefore encounter insufficient geometric constraints.

We investigated three changes while preserving frontend association, RGB-D
binding, uncertainty weights, solver limits and reconstruction settings:

| Policy | Change relative to original `native` |
| --- | --- |
| `sqrt` | Square-root camera retirement, including rank-deficient eliminated states |
| `turnover` | `sqrt` plus point retirement after two consecutive missed keyframes |
| `persistent` | Original marginalization with seed preference for longer accepted frontend histories |
| `persistent_turnover` | Persistence selection together with `turnover` |

Existing graph identities retain admission priority. Persistence selection ranks
new candidates by accepted frontend observation count within each existing 8×6
image cell, followed by spatial round robin. It uses history available at that
instant. Historical RGB-D samples do not become additional optimization factors.
The two-miss horizon and ranking rule were fixed before candidate score inspection.
Persistence selection was motivated by the first lifecycle results and evaluated
as a separate, subsequent experiment.

The structural changes are useful building blocks. Their mixed external results
do not justify replacing the validated estimator or selecting a policy per scene.
Persistence admission runs serially inside each experimental process. A production
strategy API and concurrent ingestion remain open work.

## Square-root factor contract

`slam.marginal` can eliminate cameras, landmarks or both. It consumes only raw
factors incident to retired variables and the existing historical prior. Every
other raw factor remains outside the new prior. Retiring a landmark can create
camera-to-camera information and correlations with remaining points. It does
not drop the landmark's observations or replace them with exact pose anchors.
Graph mutation occurs only after successful elimination.

For frozen weighted residual `r + J_e δe + J_k δk`, the implementation projects
onto the orthogonal complement of the observable column space of `J_e`. A second
thin SVD compresses the retained factor. Column equilibration and singular-value
thresholds of `max(1e-12, 1e-8 * largest)` determine numerical rank. Retained
columns are scaled using their norms **before** projection. Scaling projection
roundoff as fresh information would falsely constrain eliminated directions.
Residual-only constants are omitted, preserving the gradient and curvature.

This follows the square-root elimination principle described by
[Demmel et al.](https://arxiv.org/abs/2109.02182). Our implementation uses dense
double-precision SVD. Joint optimization still forms dense normal equations.
It does not implement the paper's specialized QR pipeline or establish its
speed or single-precision robustness results.

A fixed root is conditioned before removal. Rank-deficient free variables use
orthogonal elimination without damping or an information floor. Historical
weights and Jacobians remain frozen in the existing pose charts. This remains
a local Gaussian approximation, without calibrated posterior uncertainty.

With point retirement enabled, older cameras may satisfy the local support gate
through a six-direction prior block. The incoming camera still requires the
original raw multiview count and image coverage. Every joint solve still checks
the complete undamped system's observability and condition number. A local
camera block alone cannot establish global observability.

Frontend identities remain independent of graph membership. A retired identity
can seed a fresh graph variable only from a new measurement, with a new graph
ID. This discards its future identity link to the old variable. It never reuses
the retired RGB or depth sample. This is a bounded estimation policy, without
a persistent object identity or semantic-memory guarantee.

## Evaluation protocol

The accuracy study covers the ten previous regression windows, full XYZ, Desk
and RPY, and two new 300-input Room windows. This is 5,259 distinct canonical
observations across fifteen evaluation populations. Repeated policies, cached
replays and overlapping full-sequence diagnostics add no independent samples.
The ten previous windows are regression evidence. No thresholds were fitted to
their resulting scores.

Room windows 400:700 and 900:1200 were frozen before downloading and scoring the
capture. The [official TUM description](https://webshare.cvg.cit.tum.de/g/rgbd/dataset/)
explains that Room extends the Desk office to the surrounding walls. This is an
independent recording with new views and motion, sharing the office and sensor.
It does not establish generalization to a new building or sensor family. There
is still no second ScanNet capture, and its unknown clocks still prevent timed
native graph admission. Equal ScanNet results remain frontend controls.

Every candidate replays verified canonical inputs with reference poses removed
from tracking. Sparse and matched no-optimization controls remain explicit.
Sparse controls use the established 1,500-feature budget.
Largest-segment ATE is compared only alongside loss boundaries. Reconstruction
uses the original voxel and first-chunk TSDF protocol, fixed training/target
indices and common-mask scoring. Missing training references withhold the chunk.
Target RGB and depth may enter tracking, so the targets are held out from dense
fusion only. Reference-pose fusion is a control, without independent surface truth.

## Findings and limitations

Final position ATE in millimeters is shown below. All policies retain the same
loss boundaries. RPY uses its largest segment, frames 115:694, after the shared
tracking break at 115. Other populations are continuous. `Control` disables
joint optimization while retaining the matched frontend and graph maintenance.

| Population | Sparse | Control | Native | Sqrt | Turnover | Persistent | Combined |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full XYZ | 55.1 | 41.8 | 31.9 | 31.1 | 42.6 | 71.1 | 39.2 |
| Full Desk | 75.8 | 54.1 | 59.5 | 54.1 | 57.3 | 53.4 | 45.0 |
| Full RPY, largest segment | 50.6 | 39.3 | 46.1 | 49.8 | 45.6 | 47.8 | 43.3 |
| FR2 Desk early | 17.4 | 31.8 | 20.6 | 21.1 | 26.9 | 34.0 | 29.5 |
| FR2 Desk middle | 4.8 | 10.7 | 14.4 | 14.4 | 8.2 | 10.3 | 9.4 |
| FR2 Desk late | 13.1 | 7.0 | 15.7 | 15.1 | 13.0 | 15.5 | 18.6 |
| FR3 Office early | 37.3 | 12.1 | 15.5 | 15.5 | 16.7 | 19.0 | 17.9 |
| FR3 Office middle | 16.3 | 16.8 | 7.7 | 7.7 | 7.5 | 6.0 | 6.9 |
| FR3 Office late | 21.5 | 8.8 | 10.0 | 10.0 | 16.4 | 17.2 | 13.3 |
| Structure early | 11.9 | 8.8 | 9.4 | 9.4 | 7.8 | 9.5 | 9.0 |
| Structure late | 17.0 | 13.5 | 15.9 | 15.9 | 16.4 | 14.7 | 13.0 |
| ScanNet 1000 | 38.4 | 53.1 | 53.1 | 53.1 | 53.1 | 53.1 | 53.1 |
| ScanNet 3500 | 49.8 | 56.5 | 56.5 | 56.5 | 56.5 | 56.5 | 56.5 |
| New Room 400:700 | 90.8 | 120.0 | 115.6 | 130.1 | 138.3 | 106.6 | 131.8 |
| New Room 900:1200 | 64.8 | 70.6 | 67.3 | 73.6 | 51.1 | 65.6 | 45.2 |

`Combined` denotes `persistent_turnover`. Its new Room improvement in the second
window accompanies a regression in the first. Relative motion also differs from
ATE: Room 900 one-frame translation RPE is 9.99 mm for Native and 10.14 mm for
Combined. On XYZ, Combined increases it from 6.27 to 7.99 mm and increases
root-anchored orientation RMSE from 1.33° to 1.96°. Complete final and emitted
RPE, rotation, endpoint and segment metrics are retained in `summary.json`.

Reconstruction compares seven preselected populations, with 267 surface target
views and 102 first-chunk TSDF target views. Values below are surface / TSDF
missing-coverage-capped depth errors in millimeters. Lower is better. The saved
summary also includes coverage, explained coverage and common-sample depth MAE.
Sqrt is included in the full artifact, alongside all-policy common masks.

| Population, surface/TSDF views | Control | Native | Turnover | Persistent | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full XYZ, 75/22 | 35.21/32.22 | 33.60/25.98 | 32.43/44.88 | 39.21/55.66 | 40.61/30.31 |
| Full Desk, 55/23 | 61.06/53.08 | 59.33/51.64 | 59.17/50.50 | 58.09/50.56 | 56.84/50.20 |
| Full RPY, 44/11 | 81.12/94.00 | 78.75/93.96 | 83.04/114.03 | 80.20/93.96 | 78.02/94.33 |
| FR2 Desk late, 6/withheld | 67.08/— | 68.26/— | 65.32/— | 68.43/— | 67.56/— |
| FR3 Office late, 29/23 | 42.88/25.46 | 45.43/27.42 | 44.99/28.77 | 44.20/28.72 | 45.02/30.53 |
| New Room 400:700, 29/withheld | 111.53/— | 113.07/— | 113.85/— | 110.34/— | 106.06/— |
| New Room 900:1200, 29/23 | 75.15/59.15 | 88.02/72.06 | 77.13/62.06 | 78.39/60.67 | 68.77/56.67 |

Missing training references withhold the two TSDF populations. RPY's TSDF scores
its first segment, while its largest-segment ATE scores the later segment. These
are distinct populations. Outside the seven selected reconstruction populations,
the new candidates have no new mapping evaluation. Older control maps remain in
the summary with explicit missing-policy entries.

Room 400 Combined improves reconstruction while worsening ATE against Native.
XYZ Turnover improves surface error while worsening TSDF substantially. These
results reinforce the need to evaluate multiple representations and coverage.

A separate post hoc XYZ extent check preserves the earlier 81 training views
and scores 78 reference-valid targets. Whole-extent TSDF capped error is
Native 46.07 mm, Sqrt 41.38, Turnover 86.30, Persistent 86.24 and Combined 40.91.
Sparse is 90.78 and the matched control is 72.75. Surface error is respectively
35.53, 35.41, 36.56, 44.90 and 38.68 mm. Combined improves the long TSDF while
worsening its short TSDF and surface scores. Keep this enlarged-submap diagnostic
separate from the primary bounded-map populations.

Sixty-frame XYZ interventions retain the original fixed ten-frame outage and
50% depth-scale bias. Every local policy bridges missing depth with zero losses.
All policies lose eleven transitions with black RGB. Sparse loses ten with
missing depth. This robustness comes from local frontend memory and does not
establish a new optimizer benefit. Native / Persistent / Combined clean ATE is
7.68/7.27/9.69 mm and biased-depth ATE is 12.16/12.66/12.47 mm. Continuous tracking
does not identify corrupted metric depth. Fragmented black-RGB ATE is not pooled
with continuous full-sequence ATE.

Freeing capacity prevents the tested disjoint-identity stream from starving the
graph or resetting it. Dense elimination tests verify the same retained Hessian
and gradient across point retirement, conditioned-root retirement and subsequent
free-camera retirement. Tests also cover null directions, unique measurement
ownership, restored frontend state and missing acquisition clocks.

Rank-aware retirement does not automatically restore observable tracking. On
full Desk, `turnover` admits every eligible keyframe and performs just one joint
update. The undamped diagnostic finds five null directions at frame 65 and
mostly six later. The remaining observable spectrum is reasonably conditioned
at that first rejection. The failure is actual lost constraint directions in
the modeled graph, rather than justification for relaxing its rank gate. The
diagnostic leaves poses exactly equal to the uninstrumented run.

Removing rank-deficiency resets can preserve a weak graph indefinitely. More
fresh points, more admitted cameras and fewer resets therefore need separate
external trajectory and reconstruction checks. Mature frontend observations
also need not provide better geometry or correct associations. Persistence is
a selection heuristic, without an independently verified match label.
Sqrt accepts only three Desk updates and essentially matches its no-optimization
control. Combined's improved Desk ATE also comes from just one accepted update.
These outcomes do not demonstrate continuously effective smoothing.

XYZ provides a further counterexample to accepting more updates as progress.
Persistent accepts 94 joint updates versus Native's 69, while ATE worsens from
31.92 to 71.11 mm and anchored orientation RMSE from 1.33° to 6.47°. A post-fit
observer leaves its saved trajectory exactly unchanged. Among the 94 accepted,
overlapping windows, reference-relative rotation RMSE between adjacent available
cameras worsens in 53, and translation RMSE worsens in 42. The mean window
rotation RMSE changes from 0.511° to 0.532°, despite the optimizer's decreasing
objective. These are descriptive checks of overlapping windows. They do not
identify a physical sensor error or prove individual false associations.

Retired landmark estimates in the experiment archive remain their last active
values. Later camera revisions do not conditionally recover their coordinates.
Their saved residuals are historical consistency diagnostics. They are not a
fully revised persistent point map. Dense reconstruction continues to rebuild
from raw depth and final poses, independent of those sparse archive points.
Selected residual populations also differ between policies, so their raw costs
and residual summaries cannot serve as matched external accuracy rankings.
Frontend points outside the optimizer still follow their latest revised observer,
as in the original architecture. That approximation can conflict with older
observations. This study does not isolate it from selection, association or
measurement-model error as the physical cause of the real-data regressions.

The architectural priority is observability-aware local mapping with enough
verified overlap across viewpoint changes. Larger sparse systems, covisibility
selection and replayable submaps are promising follow-ups. Their benefit remains
to be demonstrated. Independent association checks, acquisition timing and
correlated depth/calibration error remain prerequisites. Global loop factors
would require separate place verification and consistent revision of geometry,
semantic attachments and spatial-memory identities.
Stable memory entities will need source-frame lineage and revision versions,
independent of ephemeral optimizer track IDs. Conditional recovery of retired
landmarks and explicit submap/gauge uncertainty remain open architectural work.

## Reproduction and evidence

Serial raw 80-frame XYZ timing on an i7-13700HX, in opposite policy orders:

| Policy | Forward/reverse fps | Keyframe p95, ms | Median joint solve, ms |
| --- | ---: | ---: | ---: |
| Native | 3.33/3.31 | 681/710 | 99/91 |
| Sqrt | 3.35/3.37 | 676/681 | 96/93 |
| Turnover | 3.21/3.21 | 761/725 | 129/132 |
| Persistent | 3.36/3.08 | 665/760 | 92/92 |
| Combined | 3.18/3.18 | 750/713 | 124/129 |

The final batch runs after all other experiments and audits, with warm files and
one policy per phase. These short shared-machine measurements have order variance
and do not establish small speed differences or real-time operation. Estimator
state stays within eight cameras and 128 points across all evaluated candidates.
The separate output archive still grows with recording length.

All 883 Linux tests pass, alongside Ruff and whitespace checks. Dense information
equivalence, nullspace preservation, streamed marginalization, outages and
ownership contracts are covered. The original full XYZ/Desk/RPY trajectories,
emitted poses and measurement archives reproduce exactly. New persistence runs
match raw/cached trajectories, measurements and non-timing diagnostics exactly,
including emitted prefix 80 and sealed prefix 40 against full XYZ. No-optimizer
turnover equals the matched no-optimizer control exactly. All 2,092 earlier
incremental-study files and 1,927 older protected scientific files were unchanged
before output compaction.
Dependencies, lockfile and defaults are unchanged. Native Windows and hosted CI
were not run.

The original commands and baselines remain available. New standard policies use
the same verified driver:

```bash
uv run python experiments/run_incremental_slam.py \
  --baseline-root outputs/local_map_reliability \
  --output-dir outputs/slam_lifecycle/regression \
  --variants sqrt turnover --workers 3

uv run python experiments/run_slam_persistent.py CASES.json \
  --output-dir outputs/slam_lifecycle/persistence_verified --workers 3

uv run python experiments/slam_lifecycle_study.py maps BASELINE CASE_ROOT \
  --variants persistent persistent_turnover

uv run python experiments/diagnose_slam_observability.py \
  --baseline-run BASELINE --features FEATURES.npz \
  --output-dir FRESH_DIAGNOSTIC --variant turnover

uv run python experiments/slam_lifecycle_study.py summary \
  outputs/slam_lifecycle/case_paths.json outputs/slam_lifecycle/summary.json

uv run python experiments/slam_lifecycle_study.py verify outputs/slam_lifecycle
```

`CASES.json` is a list of `[name, baseline_directory, features_path_or_raw]`.
Choose fresh output directories. On constrained systems, use fewer workers.
Set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` for these reference runs.
Concurrent and cached runs assess accuracy, without a latency ranking.
Serial raw timings use the `timing BASELINE OUTPUT` subcommand after other jobs.

The extent diagnostic uses the existing `experiments/evaluate_slam_extent.py`
with the same baseline and `--case-root`, a fresh `--output-dir` and the selected
`--variants`. The post-fit observer supports `--variant persistent` to reproduce
the full XYZ update audit. Outage controls use `outages SOURCE OUTPUT`, with
`--persistent` for the two persistence candidates.

Compact evidence for historical `outputs/slam_lifecycle/` now lives in
`outputs/evidence/slam_lifecycle.tar.xz`. Dataset images, feature caches and
geometry require regeneration as described in the [storage guide](output_storage.md).
The initial and subsequent persistence protocols, case manifests, archive/table hashes, paired controls,
raw/cached repeats and preserved-baseline audit accompany the results. The
`development/` runs changed source during execution and were rejected by the
driver. The first `persistence/` attempt failed on an empty selection. These
incomplete attempts are excluded. `persistence_verified/` is authoritative.
`serial_timing_final/` contains the primary runtime batch. The earlier
`serial_timing/` batch is excluded from performance conclusions because a
read-only evidence audit overlapped part of it. `excluded_timing.json` records
that distinction. `verification.json` and the final evidence manifest preserve
the retained output identities. The first observer-only diagnostic helper is
retained with its exact report hash in `diagnose_observability_v1.py`.
