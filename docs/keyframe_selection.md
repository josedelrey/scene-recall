# Motion-budget keyframe experiment

`experiments/compare_keyframe_selection.py` compares the existing temporal
keyframes with one optional motion-based policy. It reuses the current voxel
fusion, source verification, renderer and geometric diagnostics. Temporal
selection remains the production default. The preceding support-aware pruning
experiment is disabled in both arms.

## Selection rule

The optional API is `select_motion_keyframes` in
[`mapping/keyframes.py`](../src/scene_recall/mapping/keyframes.py). It accepts
only ordered candidate IDs and estimated camera poses, an exact observation
budget and a rotation lever arm. It does not accept RGB, depth or reference poses.
Between successive eligible candidates, accumulate:

```text
travel = norm(t_next - t_previous) + radius * angle(R_previous.T @ R_next)
```

Select the nearest candidates to equally spaced cumulative-travel quantiles.
Preserve the first and last candidate, resolve ties toward the earlier frame,
and constrain each choice to leave enough distinct candidates for the remaining
budget. Stationary chunks fall back to uniform candidate-index spacing. Motion
uses camera-center translation and the SO(3) rotation angle in radians.

The rotation lever arm is fixed at **1 m** for both backends and sequences,
before evaluation. It represents a nominal metric displacement per radian,
without estimating scene depth or fitting a sensor model. There is no motion
threshold or sequence-specific parameter sweep. The CLI exposes the radius
for reproducibility, rather than selecting it from held-out scores.

The rule allocates views according to estimated travel rather than elapsed
frames. It can undersample stationary periods, oversample noisy motion or
repeated traversals, and does not explicitly optimize unique scene coverage.
An overlap-aware selector would require candidate depth sampling and another
visibility/scoring policy. This experiment keeps overlap as a diagnostic to
avoid expanding a single motion-policy comparison into a second optimizer.

## Frozen comparison

The runner starts from the saved offset-five
[held-out baseline](heldout_mapping_validation.md), freezing target identities,
chunk extents, segment boundaries and exclusions before selection. Candidate
frames are all non-target observations inside each original chunk. Even targets
outside chunk extents are reserved. Held-out poses, RGB and depth never enter
selection. Held-out depth enters only evaluation after all selections are fixed.
As in the baseline, exported odometry already processed held-out observations.
This tests withheld fusion views, without claiming unseen tracking.

Both strategies retain each baseline chunk's exact keyframe budget and endpoints:
59 for Desk and 13 + 59 for RPY. Endpoint preservation fixes the map gauge and
temporal extent. No segment origins are joined. Selection uses each backend's
own estimates. Sparse and Hybrid can therefore choose different observations.
GT diagnostic maps use the corresponding selected IDs with reference poses
after selection, rather than choosing frames using reference motion.

Mapping retains 2 cm voxels, stride four, 0.3–4 m depth, 5 cm depth-edge
filtering, equal-keyframe weighting, 64-keyframe submap bounds and 10 mm discs.
Counts of sampled surface points are also retained because equal observation
counts do not imply equal valid-depth populations.

Every temporal map archive must reproduce its saved baseline exactly. Every
evaluated temporal prediction must match within 1e-12 m absolute tolerance.
Canonical source hashes, odometry identity and sampled target depths are
verified before staged outputs are published. Existing outputs are refused.

## Metrics and interpretation

All predictions use identical filtered target pixels. One exact finite
intersection across temporal/motion, Sparse/Hybrid, estimated/reference target
poses and their GT diagnostic maps supports paired MAE/RMSE comparisons.
Reports also retain the original saved common mask with explicit missing
counts. These masks are never chosen from residual size.

All-valid-target coverage and explained fractions within 2, 5 and 10 cm expose
missing geometry. Missing-penalized losses charge missing predictions the full
cap at 5, 20 and 100 cm, alongside untrimmed conditional error. Reference-target
scores test geometry at an external pose. Estimated-target scores include
correlated map/localization error. GT fusion shares depth and calibration and
is not independent surface truth.

Training diagnostics project selected source depth into selected target views
in both directions at keyframe offsets 1, 3 and 9. They report valid-depth
overlap, agreement within 5 cm and untrimmed depth RMSE under estimated and
reference motion. The chosen pairs and their temporal gaps differ by strategy,
so these are descriptive consistency diagnostics rather than paired accuracy.
Map diagnostics additionally compare both estimated maps against the same
fixed temporal GT cloud, reporting both directions of capped 2 cm surface
distance and coverage. These also share source-depth limitations.

Per-view scores and seeded ten-view block comparisons accompany pooled metrics.
The bootstrap describes sensitivity to correlated temporal blocks, without a
cross-scene independence claim. RPY retains the original unavailable second
chunk even if a changed selection avoids a missing training reference. This
prevents changing the evaluated population in favor of the new strategy.

## Desk results

Measured on 2026-10-18 with 59 keyframes per arm, 57 held-out views and 769,795
valid target pixels. The shared intersection across every variant has 713,659
pixels. Arrows denote temporal → motion. MAE is paired on that intersection,
while coverage, explained-within-5-cm and missing-penalized loss use all valid
targets. These temporal MAEs differ from older reports because the shared mask
now includes the new motion predictions.

| Map / target pose | Paired MAE (mm) | Coverage (%) | Explained (%) | L_0.2 (mm) |
| --- | ---: | ---: | ---: | ---: |
| Sparse / reference | 81.58 → 87.31 | 97.60 → 97.69 | 64.59 → 61.10 | 58.29 → 61.89 |
| Hybrid / reference | 107.23 → 108.27 | 97.30 → 97.54 | 50.76 → 50.33 | 71.61 → 72.05 |
| Sparse / estimated | 113.65 → 124.48 | 99.19 → 99.38 | 46.84 → 42.38 | 78.75 → 84.25 |
| Hybrid / estimated | 99.59 → 102.12 | 99.00 → 99.18 | 50.58 → 49.93 | 67.00 → 67.93 |
| GT / Sparse-selected | 67.72 → 74.06 | 98.87 → 99.06 | 68.30 → 68.30 | 48.51 → 50.26 |
| GT / Hybrid-selected | 67.72 → 78.51 | 98.87 → 99.03 | 68.30 → 64.01 | 48.51 → 54.23 |

Reference-target RMSE also worsens, Sparse 196.74 → 203.83 mm and Hybrid
239.30 → 240.48 mm. Sparse's reference-target MAE improves on only 10/57 views,
Hybrid's on 20/57. The ten-view block mean Sparse MAE change is +5.59 mm with a
seeded block bootstrap interval +2.58 to +8.14 mm. Hybrid's corresponding
change is +0.83 mm, interval −1.25 to +2.49 mm. These are within-sequence
dependence diagnostics, without independent-trial significance claims.

Motion selection slightly raises coverage but worsens paired accuracy and
all-target 20 cm loss. The loss of explained targets shows a failure beyond
conditional error alone. GT-pose results also worsen with the new selected
depth observations. The result cannot be attributed solely to estimated pose
error. It does not establish which physical source of multiview disagreement
or which aspect of the selected view distribution causes the regression.

| Desk estimated map | Voxels | Selected surface points | Neighbor overlap (%) | Neighbor agreement (%) | Neighbor RMSE (mm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sparse temporal | 124,493 | 796,960 | 73.94 | 91.65 | 133.72 |
| Sparse motion | 143,413 | 789,283 | 72.58 | 86.59 | 154.58 |
| Hybrid temporal | 104,802 | 796,960 | 74.72 | 95.60 | 125.22 |
| Hybrid motion | 116,137 | 790,074 | 73.50 | 94.26 | 141.83 |

Overlap is the fraction of projected source samples with valid filtered target
depth. Agreement is conditional on that overlap, using estimated relative
poses. The additional voxels do not imply better geometry. Against the fixed
temporal GT cloud, Sparse's forward/reverse within-2-cm agreement changes
45.15/52.74% → 43.85/57.26%. Hybrid changes 55.63/53.02% → 54.19/55.59%.
The reverse coverage proxy improves while forward agreement worsens.

There are 516 eligible Desk candidates. Motion selection shares only 6/59
Sparse and 5/59 Hybrid IDs with temporal selection. Maximum frame gaps grow
from 10 to 34/36, with median gaps 9 rather than 10. Reference-motion neighbor
agreement also falls, Sparse 91.37 → 88.77%, Hybrid 91.37 → 87.70%.
Equalizing estimated travel does not preserve depth consistency in this cohort.

## RPY results and generalization

RPY retains independent 13/59-keyframe chunks. Only the original first chunk's
11 held-out views are externally scored: 143,978 valid targets and 116,519
common pixels across all versions. The second chunk's 58 targets remain
unavailable because baseline training reference 445 is missing. Sparse motion
avoids that missing reference, while Hybrid motion selects missing reference
442. Neither difference changes the evaluated population.

| Map / target pose | Paired MAE (mm) | Coverage (%) | Explained (%) | L_0.2 (mm) |
| --- | ---: | ---: | ---: | ---: |
| Sparse / reference | 113.68 → 111.18 | 95.22 → 94.55 | 57.03 → 56.47 | 74.90 → 75.24 |
| Hybrid / reference | 85.86 → 84.57 | 92.16 → 92.36 | 58.07 → 58.28 | 69.24 → 68.93 |
| Sparse / estimated | 88.76 → 86.91 | 95.23 → 95.00 | 52.33 → 54.40 | 66.21 → 63.67 |
| Hybrid / estimated | 50.89 → 47.23 | 94.83 → 96.55 | 75.87 → 77.89 | 41.77 → 37.72 |
| GT / Sparse-selected | 74.60 → 72.72 | 95.48 → 94.98 | 70.51 → 72.37 | 52.09 → 50.22 |
| GT / Hybrid-selected | 74.60 → 73.48 | 95.48 → 94.51 | 70.51 → 70.94 | 52.09 → 52.12 |

The conditional Sparse/reference gain does not imply an improved tradeoff:
explained coverage falls and the 20 cm and 1 m missing-penalized losses worsen.
Hybrid/reference has small gains in all displayed metrics, unlike Desk. Its
reference MAE improves on 6/11 views, versus Sparse's 9/11, with only two
ten-view blocks. That is limited evidence from a small correlated population.
Neither RPY backend's first-chunk gains establish performance over the
unscored second chunk or new scenes.

Neighbor-view estimated overlap/agreement in RPY's first chunk changes
58.25/81.22% → 58.60/85.44% for Sparse and
58.79/91.86% → 57.90/92.02% for Hybrid. Sparse neighbor RMSE improves
160.69 → 143.29 mm. In the second chunk its agreement instead falls
89.13 → 87.25% and RMSE worsens 132.32 → 152.41 mm. Hybrid's second-chunk
agreement is nearly unchanged, 93.99 → 93.95%, with RMSE
142.13 → 135.84 mm. Pair populations differ as described above. Internal
consistency is not a substitute for the withheld external comparison.

**Decision: retain temporal selection as the default.** This fixed motion rule
does not provide a consistent accuracy–coverage improvement across Desk and
RPY. Keep the optional API and reproducible negative/mixed result. No parameters
were changed after seeing either sequence. The experiment does not establish
that all motion or overlap policies fail. A coverage objective, robust pose
uncertainty handling or online application may justify a future design, but
none is needed to complete this bounded comparison.

## Computational cost and limits

Selection uses estimated poses already available from odometry, without extra
RGB-D candidate processing. It takes about 17 ms per Desk backend and 14–15 ms
per RPY backend across both chunks. Candidate validation and path accumulation
are linear in candidate count. The simple constrained quantile search scans
at most candidate-count × budget entries and retains linear-sized arrays.

| Map construction with matched GT diagnostic | Temporal (s) | Motion (s) |
| --- | ---: | ---: |
| Desk Sparse | 7.10 | 8.60 |
| Desk Hybrid | 6.36 | 7.27 |
| RPY first-chunk Sparse | 2.56 | 2.37 |
| RPY first-chunk Hybrid | 2.62 | 2.92 |

These construction times include estimated and reference fusion, cloud
statistics and their surface-distance comparison. They are not isolated fusion
latency. Other pair diagnostics, fixed-reference comparisons and rendering are
timed separately. Second-chunk RPY times are not directly comparable because
reference completeness changes whether construction computes a paired map
comparison. The reports retain that distinction.

The complete comparison takes about 116 s for Desk and 101 s for RPY, with
Linux peak process RSS about 868 and 794 MiB. These totals include rereading and
hashing all inputs, sampling the union of selected frames for both strategies,
building four estimated/reference map pairs, diagnostics, exports and scoring.
They exclude the prior odometry run and do not estimate one arm's online cost.
Held-out buffers and pooled scoring grow with the run. The method is offline
and budget-aware, rather than a causal trigger with an unknown future path.
Observed timings are single-host measurements, without a real-time claim.

## Reproduction and retained evidence

Use the matching saved Sparse and Hybrid odometry runs and baseline, or create
them through the [odometry](odometry.md) and
[held-out](heldout_mapping_validation.md) workflows. For the baseline use
offset five, `--max-keyframes-per-submap 64` and `--render-radii-m 0.01`.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python \
  experiments/compare_keyframe_selection.py \
  --saved outputs/heldout/final/desk \
  --odometry-run \
    outputs/odometry/tum_cross_sequence/rgbd_dataset_freiburg1_desk_0_all_sparse \
    outputs/odometry/tum_cross_sequence/rgbd_dataset_freiburg1_desk_0_all_hybrid \
  --rotation-radius-m 1 \
  --output-dir outputs/heldout/keyframe_selection/desk
```

For RPY replace `desk` with `rpy` in the input/output paths. Repeat Desk into
`outputs/heldout/keyframe_selection/desk_repeat` with the same saved baseline,
then verify numerical reports, provenance and every NPZ array:

```bash
uv run --locked python experiments/verify_mapping_repeats.py \
  outputs/heldout/keyframe_selection/desk \
  outputs/heldout/keyframe_selection/desk_repeat
```

The separate experiment command is the opt-in path. The ordinary mapping and
held-out CLIs continue using their existing temporal policy. Other callers can
use the selection API and pass the resulting `Keyframe` list to `build_submap`.
Its caller must first restrict candidates to one connected chunk and exclude
every held-out observation.

Ignored output directories retain estimated/reference clouds for both arms,
observed depth and all rendered predictions, original and shared common masks,
selected IDs, candidate cumulative travel, frame gaps, source hashes, per-pair
consistency, map support, bidirectional surface agreement, per-view/pooled
scores, block diagnostics and cost breakdowns. No dependencies are added.

The full Desk repeat matched all **64 arrays across nine archives exactly**,
along with selections, numerical reports and input provenance. Only timings
and peak RSS are excluded from repeat comparison. Verification passed 559
tests, Ruff lint/format checks and whitespace checks. New tests cover exact
budgets, pure rotation, stationary fallback, repeated poses, gauge invariance,
held-out-pose/reference independence, chunk boundaries, target-depth isolation,
coverage accounting, missing references, repeat checks and atomic rejection of
changed inputs.

The runs used Python 3.12.3, NumPy 2.5.3, OpenCV 5.0.0, single-thread
OpenCV/OpenBLAS/OMP and the existing environment via `uv run --no-sync`.
Verified canonical SHA-256 fingerprints are
`fc7cd879a461d0daae2bd3e864f45e0f0deb6a09e3a8fc4d9af2691bc311484a`
for Desk and
`644fb102f9517aab7e89fd175217a0dba053cfa3c3197d1158dfe7eecafb663c`
for RPY. The experiment script SHA-256 is
`50dd0845478f8b404cb7d6e2daa940abc6561c2f666378c6f0ed8bf47a6a71d1`,
and the selection module SHA-256 is
`3028f3b583cad00d49f65f283bab38e60f09cc40e63249677845b76acc908aad`.
