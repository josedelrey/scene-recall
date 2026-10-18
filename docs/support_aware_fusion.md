# Training-view free-space consensus experiment

This experiment improves Desk's accuracy–coverage tradeoff but fails to
generalize to Sparse's external-pose RPY scores. Equal-frame voxel fusion and
Sparse remain the defaults. The implementation lives in
[`experiments/support_aware_fusion.py`](../experiments/support_aware_fusion.py).
Production fusion, odometry, calibration and rendering are unchanged.

## Rule and rationale

Start with the unchanged surface-voxel map. For each voxel supported by exactly
one training keyframe, project its centroid into every training view using that
map's poses. Sample native filtered depth at the nearest pixel, following the
existing camera convention. Let `a` count depths within the existing 5 cm
consistency tolerance and `c` count depths more than 5 cm behind the centroid.
The latter place the voxel in measured free space. Remove it only when:

```text
frame_support == 1 and c >= 2 and c > max(a, frame_support)
```

Behind-surface projections are occluded and supply no vote. Invalid depth,
off-image projections and points behind the camera also supply no vote.
Each distinct training keyframe contributes at most one vote. Multiframe voxels
always survive. Frame support supplies a conservative agreement floor when a
fused centroid does not project back onto its source sample.

This is a free-space rejection pass after fusion. It does not move centroids or
reweight retained samples. Single-view surfaces survive without contradictory
depth evidence. Estimated maps use only their own estimated training poses.
Reference poses enter the separately labelled GT diagnostic. Target depth,
target poses and target residuals never enter geometry selection.

The tolerance reuses `MappingConfig.consistency_tolerance_m`. Requiring two
contradicting keyframes protects against one disagreeing observation. Both
choices were fixed before scoring, with no threshold sweep or held-out tuning.
Distinct keyframes are still correlated measurements.

## Controlled protocol

Measurements on 2026-10-18 reuse the saved Sparse, Hybrid and GT
[held-out protocol](heldout_mapping_validation.md). Desk has 59 training frames,
57 offset-five targets and 769,795 valid target samples. Training selection,
2 cm voxels, stride four, depth/edge filtering, poses and 10 mm rendering discs
stay fixed. Offset three supplies 57 correlated sensitivity targets with the
same training maps. RPY retains its original disconnected chunks and exclusions.

The runner checks odometry identities and partitions, rebuilds every baseline
cloud with exact array equality, verifies sampled target depth and reproduces
every baseline render within 1e-12 m absolute tolerance. It verifies the full
canonical source fingerprint before publishing staged outputs.

Each baseline/proposal MAE and RMSE comparison uses exactly the same finite
pixels inside the original all-mode common mask. Rows can have different paired
populations, so these MAEs support within-row comparisons. Reports also retain
the original baseline common population with explicit missing counts. There is
no residual trimming or target-depth visibility gate. All-valid-target coverage
and explained fractions count missing geometry as failure.

Reports additionally charge for missing predictions through bounded loss:

```text
L_C = mean over all valid targets of min(abs(predicted - observed), C)
      with a missing prediction charged C
```

Caps are 0.05, 0.2 and 1 m. These losses supplement untrimmed errors, coverage
and explained fractions. A subset-only filter cannot gain coverage. Lower
conditional MAE alone is insufficient evidence of improvement.

## Desk results

Arrows denote baseline → consensus. Coverage and explained fractions use all
769,795 valid targets. Explained means depth agreement within 5 cm.

| Map / target pose | Paired pixels | Paired MAE (mm) | Coverage (%) | Explained (%) |
| --- | ---: | ---: | ---: | ---: |
| Sparse / reference | 724,829 | 82.43 → 77.88 | 97.60 → 97.03 | 64.59 → 65.61 |
| Hybrid / reference | 725,620 | 108.85 → 104.64 | 97.30 → 96.85 | 50.76 → 52.67 |
| GT / reference | 726,513 | 66.77 → 52.67 | 98.87 → 98.55 | 68.30 → 75.81 |
| Sparse / estimated | 726,096 | 113.79 → 92.45 | 99.19 → 98.77 | 46.84 → 52.16 |
| Hybrid / estimated | 726,595 | 100.27 → 85.27 | 99.00 → 98.67 | 50.58 → 56.10 |

The reproduced original 728,495-pixel common baseline MAEs remain
83.89 / 109.55 / 67.94 mm for Sparse / Hybrid / GT at reference target poses.
The paired table's baseline scores differ because it also requires a surviving
proposal prediction. This population change is reported rather than credited
to geometry improvement.

| Map / target pose | Paired RMSE (mm) | All-target L_0.2 (mm) | All-target L_1 (mm) | Lost predictions / previously within 5 cm |
| --- | ---: | ---: | ---: | ---: |
| Sparse / reference | 197.35 → 189.82 | 58.29 → 56.33 | 106.49 → 106.00 | 4,418 / 1,278 |
| Hybrid / reference | 242.81 → 234.51 | 71.61 → 69.46 | 130.84 → 130.90 | 3,484 / 1,341 |
| GT / reference | 168.35 → 140.86 | 48.51 → 40.83 | 76.08 → 64.66 | 2,448 / 577 |
| Sparse / estimated | 234.63 → 198.65 | 78.75 → 68.46 | 120.82 → 103.31 | 3,164 / 628 |
| Hybrid / estimated | 223.95 → 199.01 | 67.00 → 60.09 | 108.31 → 95.67 | 2,524 / 800 |

Removing contradictory foreground voxels often exposes an existing deeper
surface rather than a hole. GT changes 170,658 covered predictions while losing
2,448. Increased all-target explained fractions and lower missing-penalized
losses demonstrate an effect beyond discarding hard pixels. Real coverage is
still lost, including previously accurate predictions. Hybrid/reference slightly
worsens at the 1 m penalty, so the intervention does not dominate every cost.

Reference-target paired MAE improves on 40/57 Sparse views, 32/57 Hybrid views
and 56/57 GT views. Estimated-target MAE improves on 56/57 views for both
backends. Views are correlated and pooled scores weight them by pixel count.
These counts are descriptive, without an independence claim.

Desk removes 16,988/124,493 Sparse voxels, 10,598/104,802 Hybrid voxels and
11,367/106,815 GT voxels. Every removed voxel has single-frame support. The
pass took about 0.28–0.37 seconds per Desk map on this host for 59 training
views. Cost is O(weak voxels × training views), with per-view temporaries and
two per-voxel counters. This offline bounded-submap measurement does not
establish real-time or cross-machine performance.

## Sensitivity and decision

Offset-three targets preserve the Desk result without modifying the rule,
over 766,706 valid target samples:

| Map / target pose | Paired MAE (mm) | Coverage (%) | Explained (%) | All-target L_0.2 (mm) |
| --- | ---: | ---: | ---: | ---: |
| Sparse / reference | 81.90 → 76.95 | 97.59 → 97.00 | 64.14 → 65.52 | 58.65 → 56.16 |
| Hybrid / reference | 108.30 → 104.30 | 97.17 → 96.68 | 49.82 → 51.73 | 72.74 → 70.38 |
| GT / reference | 65.89 → 51.45 | 98.83 → 98.53 | 66.80 → 74.57 | 49.32 → 41.23 |

Hybrid/reference again slightly worsens at the 1 m cap, 132.44 → 132.54 mm.
This check reuses training geometry and is not an independent experiment.

RPY is a negative generalization check for Sparse at reference target poses.
Only the original first chunk's 11 views are scored, with 143,978 valid target
samples. The second chunk's 58 target views remain withheld because training
reference frame 445 is unavailable. No chunk origins are joined.

| Map / target pose | Paired MAE (mm) | Coverage (%) | Explained (%) | All-target L_0.2 (mm) |
| --- | ---: | ---: | ---: | ---: |
| Sparse / reference | 118.76 → 118.63 | 95.22 → 94.44 | 57.03 → 52.45 | 74.90 → 77.39 |
| Hybrid / reference | 89.84 → 88.07 | 92.16 → 91.77 | 58.07 → 58.84 | 69.24 → 68.80 |
| GT / reference | 76.81 → 67.07 | 95.48 → 94.76 | 70.51 → 72.85 | 52.09 → 48.36 |
| Sparse / estimated | 90.06 → 74.75 | 95.23 → 94.53 | 52.33 → 57.56 | 66.21 → 58.54 |
| Hybrid / estimated | 52.71 → 44.41 | 94.83 → 94.43 | 75.87 → 78.27 | 41.77 → 38.86 |

RPY's paired populations are 122,079 / 122,440 / 122,191 / 122,180 / 122,550
pixels in table order. Sparse/reference loses 4.58 percentage points of explained
targets and worsens at all three missing penalties. Its tiny conditional MAE
reduction does not represent a better tradeoff. Hybrid/reference also worsens
at the 1 m penalty, 167.12 → 167.84 mm. Better estimated-target scores therefore
do not justify promoting the rule. The results are consistent with estimated
training-view consistency rejecting geometry useful at external poses, but do
not isolate pose error from sensor, calibration or scene disagreement.

The intervention remains an opt-in experiment. Equal-frame fusion and Sparse
remain the defaults. Support does not certify correctness, and multiple views
can share pose error, calibration bias or a moving object. Protecting every
multiframe voxel intentionally leaves some inconsistent foreground geometry.
Nearest-pixel depth checks can reject real surfaces under pose error or
visibility changes. GT fusion shares the source depth and calibration, so these
scores measure observation consistency rather than independent surface truth.
Estimated-target improvements include correlated map and localization errors.
Independent scenes and a coverage budget fixed by application needs would be
required before considering a default change.

## Reproduction and evidence

Use the existing odometry runs, or recreate them with the
[odometry workflow](odometry.md). Regenerate a baseline with
`scripts/rgbd_heldout.py`, matched Sparse/Hybrid runs, offset five,
`--max-keyframes-per-submap 64` and `--render-radii-m 0.01`.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python \
  experiments/support_aware_fusion.py \
  --saved outputs/heldout/final/desk \
  --odometry-run \
    outputs/odometry/tum_cross_sequence/rgbd_dataset_freiburg1_desk_0_all_sparse \
    outputs/odometry/tum_cross_sequence/rgbd_dataset_freiburg1_desk_0_all_hybrid \
  --output-dir outputs/heldout/support_consensus/desk
```

Use fresh output directories. For offset three, use a baseline generated with
`--heldout-offset 3` and the same training/odometry settings. For RPY, substitute
the saved RPY baseline and matching trajectory pair. Unavailable statuses persist.

Repeat into `outputs/heldout/support_consensus/desk_repeat`, using the same saved
Desk baseline, then verify:

```bash
uv run --locked python experiments/verify_mapping_repeats.py \
  outputs/heldout/support_consensus/desk \
  outputs/heldout/support_consensus/desk_repeat
```

Reports retain pooled/per-view errors, coverage, threshold agreement,
missing-penalized losses, lost/changed prediction counts, exclusions,
input/code fingerprints and timings. NPZ files retain filtered clouds,
per-voxel keep masks, agreement/contradiction counts and training frame IDs.
Evidence counts are computed only for single-frame candidates. Hashed baseline
archives supply original voxel positions, observations, common masks and
predictions. Outputs are ignored artifacts under
`outputs/heldout/support_consensus/{desk,desk_offset3,rpy,desk_repeat}/`.

Desk's verified canonical input SHA-256 is
`fc7cd879a461d0daae2bd3e864f45e0f0deb6a09e3a8fc4d9af2691bc311484a`.
The experiment script SHA-256 is
`e7248a73e45a898949650bed499b8189ebc8ad3d72bb586e715ca12df0c1e40b`.
The local run used Python 3.12.3, NumPy 2.5.3, OpenCV 5.0.0, single-thread
OpenCV/OpenBLAS/OMP and the existing environment via `uv run --no-sync`.
No dependencies were installed or changed.

The full Desk repeat matched all 42 arrays across seven archives exactly,
including keep masks, evidence counts, clouds and predictions. The verifier
also matched input provenance and numerical reports, excluding timings. Tests
cover majority/tie behavior, missing and occluded evidence, pose direction,
distinct frame IDs, supported-voxel preservation, coverage accounting, withheld
reference chunks, target-depth independence, repeat checks and atomic failure
on changed inputs.
Verification finished with 541 passing tests, Ruff lint/format checks and
`git diff --check` passing.
