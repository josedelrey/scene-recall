# Held-out RGB-D map validation

The first `freiburg1_desk` held-out experiment does **not** support a general
geometric mapping advantage for Hybrid. With reference target camera poses,
Sparse explains more withheld depth and has lower error on identical pixels.
With estimated target poses, Hybrid improves pooled consistency, but its
per-view advantage varies substantially. These experiments measure different
properties. Sparse remains the default backend.

This extends the [existing mapping evaluation](rgbd_mapping.md). It reuses
`MappingConfig`, native-depth filtering and sampling, `select_keyframes`,
`build_submap`, voxel fusion, map export, strict odometry loading, canonical
input verification and the repeat verifier. No tracker, pose optimizer,
representation, dependency requirement or lockfile was changed.

## Experimental protocol

The primary experiment uses all 573 associated observations from TUM
`freiburg1_desk` and the previously saved Sparse and Hybrid trajectories.
Mapping integrates frames 0, 10, ..., 570, 572, exactly the existing 59-frame
training population. Held-out targets are frames 5, 15, ..., 565, giving 57
views. Other observations are unused in fusion and scoring, but all are read
and hashed to validate the original odometry input. The split is deterministic
and selected without geometry, residuals or reference poses.

Both backends use the same source identity, calibration, canonical RGB-D hash,
per-frame provenance, reference arrays, segment membership, training indices
and held-out indices. The runner refuses to label mismatched cohorts comparable.
It permits evaluation of a single backend against its GT-pose baseline.
Different tracking partitions require separate runs and cannot support a direct
paired backend claim through this runner.

Geometry settings are unchanged: 2 cm voxels, stride 4 on the native depth grid,
0.3–4 m observed depth and conservative 5 cm depth-edge rejection. Each
training frame contributes at most one centroid vote per voxel. Desk fits in
one submap with a 64-keyframe bound. There is no support-count pruning,
residual trimming, ICP alignment to targets or selection by evaluation score.
Reference integration uses the same training depth and filtering. It is a
**GT-pose reconstruction**, not an independently measured perfect surface.

The first training camera anchors each map. Reference alignment is fixed at
the first valid training reference, with metric scale unchanged. There is no
best-fit alignment over target poses or surfaces. The first-pose gauge is
cancelled analytically in `build_submap`, avoiding numerical cell-boundary
changes from multiplying an estimated anchor by its inverse. Controlled,
fully paired GT-pose map arrays must match exactly across backends.

The two target-pose experiments are:

| Experiment | Training map poses | Target camera pose | What it measures |
| --- | --- | --- | --- |
| Reference pose | Sparse, Hybrid or GT | GT, placed through the fixed training alignment | How well map geometry explains a withheld view in a common external pose frame |
| Estimated pose | Sparse or Hybrid | The corresponding exported estimated pose | Combined map and localization consistency, including correlated pose errors |
| GT baseline | GT | GT | Depth, calibration, scene, sampling and fusion limitations under reference motion |

Estimated trajectories were computed from **all** observations, including the
held-out RGB-D frames. This is held-out **fusion** validation. It is not unseen
input to the tracker, a test of relocalization, or an independent test of the
odometry training objective. The maps are offline reconstructions using past
and future training frames. This is interpolation among observed viewpoints,
not online prediction or generalization to a new scene.

## Rendering, visibility and populations

The renderer projects each voxel centroid as a camera-facing metric disc at
constant camera z. A z-buffer selects the nearest positive surface independently
of the measured target depth. This handles map self-occlusion and prevents
choosing a favorable farther surface. Discs can cover a pixel even if their
centers lie just outside the image. Points behind the camera or closer than
5 cm are omitted. No observed-depth visibility gate, hole filling or mesh
connectivity is introduced.

The primary disc radius is 10 mm, half the voxel width. Radii of 15 and 20 mm
are sensitivity checks, not independently tuned backend parameters. They
increase prediction coverage but can bridge holes, smear silhouettes and choose
a foreground envelope from inconsistent samples. Absolute scores depend on
this approximation. A surface-voxel map is not a watertight renderer.

All maps are evaluated at native pixel coordinates `[::4, ::4]`. Intrinsics
are scaled while preserving that lattice's zero-pixel origin. Target depth is
filtered identically before defining the valid target population. On Desk,
769,795 of 1,094,400 lattice samples survive filtering. Invalid, out-of-range,
hole-adjacent and depth-edge samples are excluded once for every model. Results
do not describe the unmeasured scene or the difficult boundaries removed by
this filter. RGB is retained and verified as input, but photometric prediction
is outside this geometric evaluation.

Reports distinguish three populations:

- **Own predictions:** each map's finite prediction on the fixed valid target
  population. Coverage is predicted / valid target. Conditional depth error
  uses predicted pixels and can be misleading when compared alone.
- **Common within a pose experiment:** the exact pixel intersection predicted
  by Sparse, Hybrid and GT for that target-pose mode. No residual test changes
  membership.
- **Common across all modes:** the exact intersection of all five predictions,
  allowing the same pixel population to compare pose experiments as well.
  The main accuracy table uses this stronger intersection.

Common masks are still coverage-selected subsets. They cannot establish
accuracy on missing pixels and should always accompany coverage and explained
fractions. They are recomputed for each radius and split, so error changes
across those settings are not comparisons on identical pixels.

For every population, reports include MAE, median, RMSE, p95 and maximum
absolute z-depth error, signed depth statistics and agreement at 2, 5 and 10 cm.
There is no error clipping or outlier rejection. At threshold tau:

```text
coverage = number of predictions / number of valid target pixels
conditional agreement = number with |predicted - observed| <= tau / predictions
explained fraction = number with |predicted - observed| <= tau / valid targets
```

Missing predictions have no invented depth error and fail the explained
fraction. Negative residuals place map geometry in observed free space.
Positive residuals put the rendered map behind the observed surface. The latter
may be a newly observed occluder, missing foreground geometry, pose error or
scene change. Neither sign proves the cause, so both are reported and retained
in accuracy statistics. Observed occlusion cannot be resolved reliably from a
surface map without independent visibility or scene information.

## Primary Desk results

All 57 held-out views have valid references. The table uses the 10 mm radius.
Common accuracy uses exactly **728,495 pixels** across all five predictions,
94.63% of valid targets. Coverage and explained fractions use all 769,795 valid
target pixels, including missing predictions in their denominators.

| Reconstruction / target pose | Coverage | Common MAE (mm) | Common RMSE (mm) | Common p95 (mm) | Explained within 5 cm |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sparse / reference | 97.60% | 83.89 | 201.28 | 341.27 | 64.59% |
| Hybrid / reference | 97.30% | 109.55 | 244.43 | 459.87 | 50.76% |
| Sparse / estimated | 99.19% | 115.16 | 238.29 | 394.71 | 46.84% |
| Hybrid / estimated | 99.00% | 101.03 | 225.89 | 380.25 | 50.58% |
| GT / reference | 98.87% | 67.94 | 172.17 | 224.47 | 68.30% |

With reference target poses, Hybrid's common MAE is **25.65 mm higher** and its
explained fraction is **13.83 percentage points lower** than Sparse's. Coverage
is similar, and identical-pixel scoring preserves the result. With estimated
target poses, Hybrid's pooled common MAE is **14.13 mm lower** and its explained
fraction is **3.74 points higher**. Correlated map/target pose errors can cancel
or amplify in this second experiment. It cannot establish superior external
geometry or localization accuracy.

At 5 cm, front/behind residual counts as fractions of all valid targets are
23.12% / 9.89% for Sparse with reference targets, 40.20% / 6.33% for Hybrid with
reference targets, and 29.46% / 1.12% for GT. Missing predictions account for
the remaining coverage gap. These categories partition the target population
together with agreement and missing samples. Large residuals persist even
with reference motion and must not be silently removed as occlusions.

The following middle view is selected by frame order, alongside the first and
last target, before inspecting error. Red is map geometry in front of measured
depth, blue is behind, gray is missing prediction and black is invalid target.
Color saturates at 150 mm for display only. Green marks common pixels.

![Held-out Desk frame 285 residuals and common-pixel mask](../assets/heldout_desk_middle.png)

Pixel-pooled scores give more weight to views with more valid samples. The
report also retains per-view results. On the all-mode common population, Hybrid
has lower reference-pose MAE on only **2 of 57** views. It has lower
estimated-pose MAE on **27 of 57**. Thus the pooled end-to-end improvement is
not a consistent view-level win.

For a descriptive dependence check, views are grouped into consecutive blocks
of ten within each segment, including the final partial block. A seeded
10,000-draw bootstrap resamples the six block means with equal block weight.
Hybrid minus Sparse reference-pose MAE averages +25.76 mm, with a block-bootstrap
95% interval of +16.60 to +36.43 mm. Estimated-pose MAE averages -19.88 mm,
with an interval of -83.03 to +39.03 mm. Block-weighted values differ from
pixel-pooled differences by design. These six blocks and arbitrary block size
provide a sensitivity description, not independent trials, a formal guarantee
or evidence of cross-sequence statistical significance.

## Sensitivity and disconnected tracking

| Disc radius | All-mode common pixels | Sparse reference MAE (mm) | Hybrid reference MAE (mm) | Sparse estimated MAE (mm) | Hybrid estimated MAE (mm) | GT MAE (mm) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 mm | 728,495 | 83.89 | 109.55 | 115.16 | 101.03 | 67.94 |
| 15 mm | 750,394 | 91.04 | 119.64 | 132.03 | 118.72 | 82.51 |
| 20 mm | 753,762 | 97.15 | 126.91 | 144.39 | 132.01 | 95.10 |

The qualitative ordering survives all three footprints. More coverage comes
with a worse foreground-envelope error, including for GT. A low-coverage,
small-footprint renderer must not be favored by its conditional error alone.

A second split keeps training unchanged and evaluates frames 3, 13, ..., 563.
Its 57 views contain 766,706 valid targets and 724,289 common pixels. Reference
pose common MAE is 83.29 mm for Sparse and 109.09 mm for Hybrid. Estimated pose
common MAE is 116.63 and 99.38 mm respectively. GT is 67.01 mm. Explained
fractions are 64.14% / 49.82% for Sparse / Hybrid with reference targets and
45.97% / 50.80% with estimated targets. These are correlated checks of split
sensitivity, not additional independent sequences.

A map never spans a tracking reset. Targets must lie strictly between the first
and last training frame of one bounded submap. Targets in gaps between chunks
are explicitly excluded, rather than assigning them to whichever map scores
best. Single-frame chunks have no interior validation targets. Reference poses
do not bridge segments. Missing or nonrigid target references withhold that
view from the controlled comparison. If any training reference is unavailable,
the entire chunk's controlled comparison is withheld, preserving estimated
fusion of all its training observations. The runner reports these losses,
keeps estimated maps, and never fabricates GT predictions.

The saved Sparse and Hybrid `freiburg1_rpy` runs both reset at frame 115.
They create independent maps for frames 0–114 and 115–693, with 13 and 59
training frames. There are 69 candidate held-out views. The first segment
scores 11 views with 143,978 valid targets and 122,884 common pixels. The second
segment's final training frame 693 has no reference, so its 58 candidate views
are explicitly unavailable for the controlled comparison. Both estimated maps
retain all selected geometry. This intentionally conservative policy avoids
comparing a complete estimated map against an incomplete GT training cohort.

On the small first RPY segment, common reference-pose MAE is 119.64 mm for
Sparse, 90.95 mm for Hybrid and 78.36 mm for GT. Estimated-pose MAE is 91.16 /
53.65 mm. Reference-pose explained fractions are 57.03% / 58.07% / 70.51% for
Sparse / Hybrid / GT. This segment favors Hybrid and further limits any claim
that either backend is generally better. It does not change the Desk result
and cannot describe the withheld second segment.

## Reproduction and retained artifacts

Run from the repository root with the saved format-2 odometry exports and the
unchanged local TUM datasets available:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python \
  scripts/rgbd_heldout.py \
  --odometry-run \
    outputs/odometry/tum_cross_sequence/rgbd_dataset_freiburg1_desk_0_all_sparse \
    outputs/odometry/tum_cross_sequence/rgbd_dataset_freiburg1_desk_0_all_hybrid \
  --render-radii-m 0.01 0.015 0.02 \
  --output-dir outputs/heldout/reproduction/desk
```

The default held-out offset is 5 and the default submap bound is 64 training
frames. Add `--heldout-offset 3` for the alternate split. Replace `desk` with
`rpy` in both odometry paths for the disconnected experiment. Use fresh output
directories. `--source-path` supports the existing source relocation policy.
The full canonical fingerprint, source and calibration must still match.

Each output contains `report.json`, `summary.csv`, selected input provenance in
`inputs.json`, estimated/reference map NPZ and PLY files, per-chunk rendered
view NPZ files, and first/middle/last target diagnostic PNGs for each radius.
View archives retain measured depth, every prediction, common masks and both
candidate and evaluated frame indices. Reports contain per-view and pooled
population counts, numerical errors, excluded statuses, map membership, input
hashes, odometry hashes, package code fingerprint and environment versions.
No partially validated run is published. Existing output is preserved.
Sampling retention is bounded by the current training chunk and its targets.
Pooled scoring loads rendered arrays across the run, so total evaluation memory
is not bounded independently of sequence length. This is an offline evaluator.

Final local outputs are `outputs/heldout/final/desk/`, `desk_offset3/`, `rpy/`
and `desk_repeat/`. Generated artifacts are ignored. This document and one
representative image retain the conclusions in Git. The source fingerprint is
`fc7cd879a461d0daae2bd3e864f45e0f0deb6a09e3a8fc4d9af2691bc311484a`.

Repeat the primary command into a new directory, then run:

```bash
uv run --locked python experiments/verify_mapping_repeats.py \
  outputs/heldout/final/desk outputs/heldout/final/desk_repeat
```

The complete three-radius Desk repeat matches exactly: seven NPZ archives,
61 arrays including NaN masks, input provenance, source/code/odometry hashes,
per-view diagnostics and pooled numerical summaries. Timing and RSS are
excluded from equality. The result is saved in
`outputs/heldout/final/reproducibility.json`. The primary run took 75.05 s with
558.7 MiB Linux peak process RSS. The alternate split took 47.60 s and the RPY
check 46.74 s. These observations include possible process contention.

The current host used the existing repository `.venv` through
`UV_CACHE_DIR=/tmp/scene-recall-uv-cache uv run --no-sync`, retaining the prior
offline environment. No packages were installed. Runtime and peak RSS include
validation, rendering, metrics, archives and diagnostics. Some runs overlapped,
so resource observations are not isolated integration benchmarks.

## Verification and remaining limits

All 523 tests pass, including 14 new held-out tests. Ruff lint and formatting
checks and `git diff --check` pass. Synthetic tests check exact analytic ray/disc
visibility against brute force, foreground z-buffer occlusion, behind-camera
rejection, inverse camera transforms, a rotated plane and the evaluation lattice
origin. Contract tests cover signed residuals, empty and missing predictions,
coverage-induced ranking reversals, common-mask identity, resets, chunk gaps,
missing target and training references, reference-grid independence from the
estimated anchor, refusal of mismatched cohorts, atomic failure on a modified
held-out input, repeated numerical identity, and unchanged map arrays when
only held-out depth changes. Existing mapping and real-adapter tests also pass.
The retained first/middle/last Desk PNGs were inspected visually. No new
interactive viewer was introduced.

The previous map-distance and cross-view measurements remain valid for their
stated populations. They showed less dispersed geometry and better relative
consistency for Hybrid on Desk. This held-out evaluation adds a different
constraint: explaining observations at externally supplied camera poses. Its
results do not support upgrading that earlier evidence into a general claim of
more accurate Hybrid maps. End-to-end improvements remain useful as a
self-consistency diagnostic and need external localization evaluation.

The main unresolved limitations are a single primary sequence, highly
correlated interleaved views, sensor/reference timing and calibration error,
dynamic objects, point-disc rendering, first-pose alignment sensitivity and
surface fusion without free-space reasoning. The nonzero GT baseline is an
empirical warning about the combined evaluation/representation floor, not a
calibrated bound on attainable accuracy. No image explains causality by itself.
A future benchmark should add independent scenes and wider temporal gaps,
retain the fixed-population and coverage reporting, and test representation or
pose changes against this baseline before expanding the mapping system.
