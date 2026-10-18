# Projective volumetric fusion investigation

This investigation adds an optional bounded TSDF reconstruction with connected
colored meshes. The original surface-voxel mapper, temporal keyframes and
Sparse default remain unchanged. The new path is useful research infrastructure,
with substantial gains under some poses and important failures under others.

## Why change the representation

Earlier controls found substantial held-out residuals even with reference
poses. More observations and finer voxels often made the nearest foreground
envelope worse. Support-aware deletion helped Desk but failed on Sparse/RPY.
Motion-budget keyframes added geometry without a consistent accuracy–coverage
gain. Together these findings favored testing surface fusion before another
odometry parameter sweep or a more elaborate keyframe policy.

The baseline averages points only after they enter the same voxel. Measurements
on opposite sides of a cell boundary remain separate layers. It also lacks a
shared free-space field or connected surface. The synthetic noisy-plane test
demonstrates this boundary effect: signed-distance fusion more than halves
point-to-plane mean error relative to the baseline on controlled training data.
That test is not a held-out scene result.

The approach builds on the signed-distance integration principle in
[Curless and Levoy, 1996](https://lightfield.stanford.edu/papers/volrange/) and
[KinectFusion, 2011](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ismar2011.pdf).
It is a small NumPy implementation, without an external reconstruction library,
learned model, loop closure or pose optimizer.

## Implementation

[`mapping/tsdf.py`](../src/scene_recall/mapping/tsdf.py) accepts training IDs,
camera-to-map poses and already-filtered sampled depth lattices. It never reads
reference or held-out observations. Estimated and reference-pose integrations
are separate calls. The comparison uses the baseline's exact stride-four depth
samples and correspondingly scaled intrinsics, rather than giving TSDF full
native-resolution depth.

At each active voxel center, project into each training view and sample nearest
depth. Let `r = measured_depth - camera_z`. With truncation `mu`:

```text
r < -mu: no observation, behind the visible surface
r >= -mu: add one vote of min(r, mu)
TSDF = sum of votes / number of votes
```

Positive values represent observed free space and negative values the narrow
region behind a surface. Every distinct keyframe has equal weight. Sorting by
frame ID makes input-list order irrelevant. No sensor-confidence law, residual
trimming, sign-dependent evaluation exclusion or reference fitting is added.

The fixed configuration is **2 cm voxels and 6 cm truncation**, three cells.
No truncation sweep or sequence-specific adjustment was performed. Allocate the
union of sampled training depth bands, dilated by one cell. Then integrate all
training views over that fixed set, allowing earlier free-space evidence to
affect surfaces allocated by later views. This is deliberately offline.
Sparse field persistence is bounded at two million active voxels. Dense scratch
indexing is separately bounded at twenty million cells. Overflow raises and
discards staged output, rather than pruning geometry.

Surface extraction averages interpolated edge zero crossings within fully
observed cells. Connect four valid neighboring surface cells around each
sign-changing grid edge, splitting quads along a fixed diagonal. Unknown corners
and missing neighbors leave open boundaries. There is no extrapolation or hole
filling. This cell-based extraction does not guarantee watertightness or resolve
ambiguous cells containing multiple surfaces.

[`mapping/mesh.py`](../src/scene_recall/mapping/mesh.py) renders triangles at
pixel centers with perspective-correct depth, near-plane clipping and nearest
positive visibility. Both face orientations render. Target depth never enters
rasterization. Binary colored PLY and full-precision NPZ meshes retain the map
gauge. Colors use only nearby training observations and never influence shape.

Volume `frame_support` counts valid projective votes, including free space.
Extracted-point `view_support` counts distinct training views within the
truncation band. These differ from baseline within-voxel frame support.
Signed residual spread is descriptive, without a calibrated uncertainty claim.

## The failed disc control

The first experiment revoxelized extracted points and rendered them with the
baseline's unchanged 10 mm discs. Desk GT coverage dropped from 98.87% to
87.90%, and common-pixel MAE worsened. This negative result is retained.

An independent exact-plane test identifies a representation mismatch: 10 mm
discs on a 20 mm point grid leave holes even when every point has exactly correct
depth. The mesh covers that plane exactly. Analytic rotated-plane, slanted
triangle and independent ray-intersection tests validate the new rasterizer.

The final experiment therefore retains three predictions for each map/pose:
original surface-voxel discs, TSDF point discs as a diagnostic, and connected
TSDF triangles. The primary comparison evaluates the complete representation
and renderer together. It does not isolate fusion from surface interpolation.
Fusion settings stayed fixed during this correction. Increasing a disc radius
to optimize held-out error was deliberately avoided.

## Controlled evaluation

[`experiments/compare_tsdf_fusion.py`](../experiments/compare_tsdf_fusion.py)
reuses the saved Sparse/Hybrid/GT fusion-holdout protocol. Desk retains its 59
training frames and 57 targets. RPY retains independent 13/59-frame chunks and
all original exclusions, leaving 11 externally comparable first-chunk views.
The second chunk's 58 views remain withheld for missing training reference 445.
Estimated geometry is still reconstructed in both chunks.

XYZ adds a third sequence using existing frozen odometry. Its 81 temporal
keyframes form 64/17-frame chunks. There are 78 eligible held-out candidates,
with target 195 withheld for a missing reference, leaving 77 evaluated views.
The temporal gap between chunks stays excluded. No disconnected origins are
joined, no new keyframes are selected and no held-out depth enters fusion.

Depth bounds, edge rejection, voxel width, training views and poses are fixed
between representations. GT fusion uses those same depth observations with
reference poses as a diagnostic, rather than independent surface truth.
Odometry has already processed the held-out observations. This remains withheld
fusion validation among observed viewpoints, rather than unseen tracking.

Every reconstructed baseline cloud must exactly match its saved archive, and
every baseline render must agree within 1e-12 m absolute tolerance. Full canonical
source fingerprints and odometry identities are verified before publication.

Paired MAE/RMSE use one exact intersection across all primary baseline/mesh
maps and target-pose modes. The incomplete disc control does not constrain that
mask. Original baseline-common populations are also retained with missing
counts. All-valid-target coverage and explained fractions within 2/5/10 cm
expose discarded or missing geometry. Missing-penalized capped losses charge
missing predictions the full cap at 5, 20 and 100 cm. No residual trimming or
target-depth visibility gate is used.

## Measured results

Measurements are from 2026-10-18. Arrows denote surface voxels → connected
TSDF mesh. Each sequence uses one exact common population across every primary
map and pose mode. Coverage, explained-within-5-cm and capped loss use all valid
targets, with missing geometry counted as failure. Common populations are
647,319 / 91,050 / 904,371 pixels for Desk / RPY / XYZ, from
769,795 / 143,978 / 1,046,358 valid targets. The shared-mask baseline MAEs
therefore differ from earlier reports.

| Sequence / map at reference target pose | Paired MAE (mm) | Paired RMSE (mm) | Coverage (%) | Explained (%) | L_0.2 (mm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| DESK / Sparse | 66.61 → 124.47 | 155.10 → 250.23 | 97.60 → 92.38 | 64.59 → 39.64 | 58.29 → 92.45 |
| DESK / Hybrid | 89.26 → 78.93 | 193.60 → 201.63 | 97.30 → 93.80 | 50.76 → 66.23 | 71.61 → 60.88 |
| DESK / GT | 57.34 → 30.60 | 134.50 → 102.78 | 98.87 → 96.95 | 68.30 → 87.23 | 48.51 → 30.66 |
| RPY / Sparse | 71.69 → 70.45 | 180.95 → 153.96 | 95.22 → 83.32 | 57.03 → 45.56 | 74.90 → 86.78 |
| RPY / Hybrid | 54.39 → 54.73 | 133.98 → 132.22 | 92.16 → 85.18 | 58.07 → 55.56 | 69.24 → 76.32 |
| RPY / GT | 41.33 → 28.64 | 128.64 → 103.96 | 95.48 → 84.98 | 70.51 → 71.36 | 52.09 → 55.01 |
| XYZ / Sparse | 52.34 → 94.43 | 148.20 → 209.51 | 98.34 → 95.21 | 78.10 → 49.89 | 41.34 → 72.19 |
| XYZ / Hybrid | 45.48 → 47.24 | 146.41 → 160.57 | 97.49 → 95.90 | 80.77 → 81.03 | 36.78 → 37.97 |
| XYZ / GT | 44.59 → 20.93 | 141.78 → 90.22 | 98.57 → 97.16 | 81.68 → 91.79 | 34.54 → 21.24 |

The strongest controlled gain is GT-pose fusion on Desk and XYZ. Both lower
paired MAE, RMSE and all-target missing-penalized error while explaining more
target geometry. Desk loses 1.92 coverage points and XYZ loses 1.42. RPY is a
counterexample: GT conditional accuracy improves, but coverage falls by 10.50
points and L_0.2 worsens. A conditional accuracy gain alone is insufficient.

Sparse reference-target geometry regresses badly on Desk and XYZ. On RPY its
small common-pixel MAE gain also comes with worse all-target costs. Hybrid has
a Desk MAE/explained-fraction gain, but RMSE and the 1 m missing penalty worsen.
Its XYZ and RPY external scores do not establish a reliable improvement.

| Sequence / estimated target pose | Paired MAE (mm) | Explained (%) | L_0.2 (mm) |
| --- | ---: | ---: | ---: |
| DESK / Sparse | 101.58 → 64.72 | 46.84 → 67.40 | 78.75 → 55.48 |
| DESK / Hybrid | 86.88 → 57.19 | 50.58 → 78.20 | 67.00 → 43.38 |
| RPY / Sparse | 61.57 → 39.86 | 52.33 → 63.92 | 66.21 → 62.67 |
| RPY / Hybrid | 31.33 → 18.74 | 75.87 → 80.92 | 41.77 → 42.15 |
| XYZ / Sparse | 98.98 → 46.84 | 47.85 → 76.74 | 73.35 → 41.94 |
| XYZ / Hybrid | 36.28 → 18.22 | 85.20 → 93.09 | 29.24 → 18.90 |

Estimated-target MAE improves in every displayed case, often substantially.
This is map/localization self-consistency under correlated poses. It cannot
override the external-pose regressions. RPY/Hybrid estimated L_0.2 even worsens
slightly despite lower paired MAE and more explained targets, because missing
predictions also matter. Large-error/missing penalties are retained at all
three caps in the reports.

GT common-pixel MAE improves on 50/57 Desk views and 70/77 XYZ views. Seeded
ten-view block mean changes are −26.92 mm (bootstrap interval −38.61 to −13.12)
and −23.72 mm (−28.45 to −19.50), respectively. RPY has only 11 externally
comparable views and two blocks. These are dependence diagnostics within the
same captures, rather than independent-scene confidence claims. Sparse's
reference-target MAE improves on only 4/57 Desk and 18/77 XYZ views.

## Pose-bias diagnostic and decision

[`experiments/diagnose_tsdf_pose_bias.py`](../experiments/diagnose_tsdf_pose_bias.py)
holds an exact stationary plane and observed target fixed. Keep the first
training pose correct, and shift eight later training estimates by 8 cm along
the optical axis. Fusion removes the geometrically correct minority layer as
free-space evidence accumulates under biased poses. On the fixed principal ray:

| Synthetic condition | Baseline reference error | TSDF reference error | Baseline estimated-target error | TSDF estimated-target error |
| --- | ---: | ---: | ---: | ---: |
| Exact training poses | 0 mm | 0 mm | 0 mm | 0 mm |
| Eight biased training poses | 0 mm | 80 mm | 80 mm | 0 mm |

The report also retains a central image region, where point footprints change
which layer wins. The principal ray is fixed analytically, independent of any
residual. This is a causal counterexample explaining how self-consistency can
improve while external geometry regresses. It does not estimate the real pose
bias or isolate calibration, exposure timing, dynamics and tracking drift in
the captured sequences.

**Keep the current defaults.** The new representation demonstrates substantially
better fusion when reference poses are supplied, especially on Desk and XYZ.
It also improves estimated-target consistency. Neither establishes better
end-to-end Sparse reconstruction. RPY's coverage loss and Hybrid's mixed tails
prevent a general default recommendation. The original baseline and failed
point-disc control remain reproducible. No algorithm parameters were tuned
after the first real-scene run. The connected renderer was added following the
independent analytic coverage failure, with the same TSDF field and extracted
point controls retained exactly.

The next accuracy priority is training-only multiview pose constraints and
pose/fusion interaction, with a fixed gauge and conservative RGB/depth agreement
checks. Depth-objective improvement alone was already unreliable for Hybrid,
so a new optimizer needs external-pose map validation and synthetic pose/noise
controls. Independent scenes and an independent surface reference would then
test generalization. Block-sparse storage and faster raycasting are efficiency
priorities after those accuracy questions. Semantic association and loop closure
should build on validated geometry and retained observation provenance.

## Cost and practical limits

On the first serial connected runs, Desk estimated TSDF field integration takes
about 2.5 s per map. Mesh extraction adds about 0.34 s. Revoxelized control-point
extraction, coloring and depth-support diagnostics add another 0.48–0.50 s.
The repeat records higher construction times, so these are descriptive host
measurements rather than a speed guarantee. Baseline construction timings also
include a reference map and surface-distance diagnostics, making them unsuitable
for a direct fusion-latency speedup claim.

Complete comparison runtime is about 99 / 77 / 107 s for Desk / RPY / XYZ, with
peak Linux RSS about 777 / 952 / 807 MiB. Totals include source rereading and
hashing, baseline reconstruction, three TSDF fields, both TSDF render controls,
exports, scoring and diagnostic images. Prior tracking is excluded. No GPU or
new package dependency is required. Processing remains offline, with per-chunk
fusion bounds and run-wide held-out scoring buffers.

The z-depth TSDF is projective, using nearest sampled pixels. The fixed narrow
band assumes useful pose consistency, and full-cell visibility requirements
can remove real geometry, especially with sparse overlap or pose disagreement.
Free-space votes can erase real surfaces under systematic pose error. Thin
structures, dynamic objects, range-dependent noise, calibrated confidence and
surface ambiguity remain unresolved. The model is neither a SLAM system nor a
guarantee of persistent semantic geometry.

## Reproduction

Use the saved odometry and baseline runs, or regenerate them through the
[odometry](odometry.md) and [held-out](heldout_mapping_validation.md) workflows.
Baselines use offset five, `--max-keyframes-per-submap 64`, and
`--render-radii-m 0.01`.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python \
  experiments/compare_tsdf_fusion.py \
  --saved outputs/heldout/final/desk \
  --odometry-run \
    outputs/odometry/tum_cross_sequence/rgbd_dataset_freiburg1_desk_0_all_sparse \
    outputs/odometry/tum_cross_sequence/rgbd_dataset_freiburg1_desk_0_all_hybrid \
  --truncation-m 0.06 \
  --output-dir outputs/heldout/tsdf/connected/desk
```

For RPY substitute `rpy` in the paths. XYZ odometry lives under
`outputs/odometry/tum_freiburg1_xyz/`. Generate its baseline with the standard
held-out command before running this comparison. Use fresh output directories.

Repeat Desk into `outputs/heldout/tsdf/connected/desk_repeat`, then run:

```bash
uv run --locked python experiments/verify_mapping_repeats.py \
  outputs/heldout/tsdf/connected/desk \
  outputs/heldout/tsdf/connected/desk_repeat
```

The runner retains fields, meshes, colored PLY, point controls, sampled-input
provenance, predictions, common masks, per-view/pooled metrics, paired temporal
blocks, point-sampled bidirectional map proxies, resource breakdowns and three
residual panels per evaluated chunk. The point-distance proxy samples the TSDF
surface, rather than computing an exact point-to-triangle distance. Changing
surface sampling changes that proxy, so it is secondary to held-out depth.

Retained final outputs are under
`outputs/heldout/tsdf/connected/{desk,rpy,xyz,desk_repeat}/`. Initial disc-only
outputs remain under `outputs/heldout/tsdf/{desk,rpy,xyz}/`. Every original
prototype field, point and prediction array matches the corresponding retained
control exactly after adding the mesh renderer, excluding the deliberately
changed common mask. The full Desk repeat matches **64 arrays across ten
archives** plus numerical reports and provenance, excluding runtime/RSS.

Run the synthetic diagnostic into a fresh directory:

```bash
uv run --locked python experiments/diagnose_tsdf_pose_bias.py \
  --output-dir outputs/heldout/tsdf/pose_bias_verified
```

The existing environment used Python 3.12.3, NumPy 2.5.3, OpenCV 5.0.0 and
single-thread OpenCV/OpenBLAS/OMP through `uv run --no-sync`, without installs.

Final verification passes **581 tests**, Ruff lint/format checks and whitespace
checks. Tests include signed-distance denoising, pose direction, sorted-frame
reproducibility, free-space/occlusion semantics, unknown-cell boundaries, color
independence, resource overflow, target-depth isolation, missing references,
atomic failure, analytic rasterization and the synthetic pose-bias reversal.
