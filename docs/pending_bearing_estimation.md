# Persistent pending bearings in shared RGB-D estimation

The experimental shared estimator can retain measured RGB rays before a feature
has a frontend 3D owner, valid native depth or initialized metric range. Verified
native-anchor associations can connect those rays to depth received later.
An optional multiview path initializes finite range when the retained geometry
supports it. The default shared estimator remains unchanged.

## Evidence before state initialization

The preceding [geometric-support investigation](observable_shared_estimation.md)
identified RPY cameras with hundreds of image features but very few valid native
depth bindings. The shared ledger previously captured only frontend-owned
features and admitted new identities only with native depth. Increasing the
observation budget could not recover observations omitted before admission.

`--shared-bearings retain` captures all extracted image-feature candidates before
that owner filter. Admission first applies the legacy selection to currently frontend-owned rows,
then adds up to 128 spatially distributed rows. Subsequent causal tracking and
identity changes can change the legacy selection itself. The additional budget is
controlled by `--shared-bearing-budget`. Native sample ownership is unique across
the combined packet. A duplicate native owner is skipped, with candidate and
retained counts kept distinct. This policy changes the retained evidence budget,
so it is a separate control from association and triangulation.

Each admitted row keeps its camera and feature ID, pixel, immutable exposure
transform, optional native sample and whitening, raw identity and descriptor.
Unowned frontend features have distinct negative provenance keys and positive
ledger track IDs. These keys never become frontend metric landmark IDs.
Unknown world-point seed metadata consists of three NaNs. `RawObservations`
allows this explicit pending state. Ordinary `DirectFactors` and every solver
factor still require finite initialized metric points.

Recording an observation does not create a landmark state or a pose constraint.
Pending-only cameras retain explicit output attachments until usable observations
connect them. Their raw components and fixed unused camera variables remain
visible in diagnostics. The first pending packet cannot take over publication's
first metric gauge. Connected output coverage includes the existing attachment
policy and is not a global observability certificate.

## Reversible native-anchor bridges

`--shared-bearings anchor` additionally tests up to three recent retained cameras
inside the existing same-segment historical exclusion interval. The route cannot
replace cross-segment loop confirmation. Registration uses only owned retained
native samples, transformed with their actual frozen RGB/depth exposure, and
owned RGB observations. Frontend 3D point seeds do not enter registration.

Mutual descriptor correspondences are filtered to retained rows. A native-anchor
PnP hypothesis fits even correspondences and is verified on odd correspondences,
with image, available depth, coverage and metric-geometry gates. The opposite
native-anchor direction is tried if the first direction fails. Withheld rows do
not fit the hypothesis or propose those anchor identity unions.

Accepted fitting rows propose reversible landmark equivalences. A ray lacking
source depth can also attach to a later target-native sample by checking its
backward projection under the verified transform. The associated transform
initializes placement between old gauges and never adds a pose likelihood or
replaces an odometry measurement. Every RGB and native likelihood remains owned
by its original observation. Registration verification is a selection procedure,
not an independent sensor likelihood or association-correctness guarantee.

Revocation removes the proposal, rebuilds the live partition from raw packets,
invalidates affected point seeds and derivatives, and republishes tracking and
native reconstruction consistently. A raw packet's fingerprint does not change
when its identity class becomes initialized or is split again.

## Deferred multiview range

`--shared-bearings multiview` also permits fitting correspondences with no native
depth on either side when they pass a Sampson pixel gate under a separately
verified native-anchor transform. At effectively zero translation, a rotation
projection gate replaces the undefined epipolar line. These associations can
remain pending. They do not themselves certify finite depth.

A class without native depth is eligible for Euclidean initialization only with
at least three views inside one already metric-connected raw camera component.
For world ray directions d_i and camera centers c_i, the initial point solves

```math
\left[\sum_i (I-d_i d_i^T)\right]X
=\sum_i (I-d_i d_i^T)c_i.
```

Initialization requires at least one degree of maximum ray parallax, a full-rank
ray information matrix at the recorded tolerance, positive camera depth in every
view and at most three pixels of reprojection error in every view. No depth or
pixel is synthesized or rewritten. Camera placement is estimated, so these gates
establish conditional geometric eligibility, not calibrated range uncertainty.
A pending class cannot use arbitrary placement between independent metric gauges
to manufacture a scale or a connection.

After initialization the ordinary shared camera-landmark objective uses every
retained row in that class. For color-camera pose T_i, native-to-color exposure
E_i, native measurement z_ij and pixel u_ij, its terms remain

```math
\sum_{j\in\mathcal I}\sum_{i\in\mathcal O_j}\left[
\rho_R\!\left(\|\pi(T_i^{-1}X_j)-u_{ij}\|^2\right)
+\mathbf 1_{z_{ij}\;\mathrm{owned}}
\rho_D\!\left(\|W_{ij}(E_i^{-1}T_i^{-1}X_j-z_{ij})\|^2\right)\right].
```

The initialized set I is explicit. Pending rows stay in the raw ledger outside
this metric solve. Landmark elimination remains temporary and preserves the
coupled camera information. There is no retained damping prior or second local
objective. Existing valid point estimates warm-start an unchanged partition.
Identity edits rebuild seeds from raw depth or currently eligible multiview rays.

This implementation chooses deferred Euclidean initialization to keep the
complete Cartesian Schur solver as a direct scientific reference. [Inverse-depth
representations](https://webdiis.unizar.es/~josemari/InverseDepth_17_april_final_www.pdf)
are an important alternative for representing distant features before finite
range becomes observable. They would require a different state chart and careful
elimination and gauge tests. Pure rotation and points at infinity remain pending
here, so useful rotation-only bearing information is still not fully exploited.

## Configuration and saved provenance

All modes require `--shared-estimation`. The four modes are `off`, `retain`,
`anchor` and `multiview`, with `off` as the unchanged default. For example:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python scripts/rgbd_slam.py \
  --tum-path local_inputs/recording --output-dir outputs/bearing_run \
  --shared-estimation --shared-bearings anchor --shared-bearing-budget 128 \
  --contrast clahe --feature-scale 0.5 --flow-recovery --map
```

The saved shared archive retains the original raw arrays and identity-owner
records. When descriptors are available for every recorded view, the export includes
aligned `bearing_descriptors`. It also records `bearing_edges` with the selected native-anchor direction, and `bearing_seed_ids`
and `bearing_seed_points` for pending classes admitted to the metric objective.
These initialization seeds support frozen replay even if a later numerical
rollback clears the optimized-point cache. They are not additional measurements.
The graph audit and frozen-factor replay include initialized image-only classes.

`experiments/bearing_shared_study.py` freezes policy configuration, worker source
snapshots, driver hashes and source input ranges. It provides retention,
association, multiview and support-combination controls. The complete shared
solver remains the reference. A fresh output directory is required.

## Evaluation design

Promotion criteria are recorded before the main real-recording results in
`outputs/bearing_shared/promotion_protocol.json`. They require a substantial
matched full-RPY trajectory improvement with preserved coverage, improved
common-training reconstruction, acceptable reverse support, transfer reliability
and the existing ownership, causal and exact-refusion contracts. These are
exploratory research gates, not calibrated downstream scene-memory requirements.
Application tolerances still need validation against object localization and
retrieval tasks.

The main matrix uses complete TUM Desk573, RPY694 and XYZ792, ScanNet160,
ETH Room263 and Sofa533. Controls reuse protected complete-solver artifacts.
Early short runs were software-development pilots, including a subsequently
corrected pending-camera publication issue. Their source snapshots are retained
and their results are excluded from promotion evidence.

An independent [OpenCV RGB-D odometry](https://docs.opencv.org/4.13.0/d0/d60/classcv_1_1rgbd_1_1RgbdOdometry.html)
control runs in an isolated OpenCV contrib environment. It uses fixed CLAHE,
static calibration to register native depth into RGB, a 0.3 to 4 meter depth
range, and the library's odometry settings. Its synthetic translation control
checks the source-to-target convention before recordings are processed.
It receives identical canonical inputs, with references reserved for scoring,
and reconstructs from the same original native samples. It has no loop closure
or historical optimization, so it provides an external odometry control rather
than a comparison with a complete established SLAM system. Environment versions,
input hashes and exact reconstruction checks accompany its results. Core project
dependencies remain unchanged.


## Causal trajectory and coverage

The main matrix completed all three candidate policies on six recordings and
3,015 distinct received inputs. Each comparison uses the intersection of the
candidate and reference final root components, then counts only frames with
valid reference poses. Whole-root error uses the candidate's own connected
population. Pose alignment fixes scale and anchors the first scored root pose.
These results come from exploratory deterministic recordings.

| Recording | Policy | Connected / received | Matched scored frames | Reference matched RMSE, mm | Candidate matched RMSE, mm | Candidate whole-root RMSE, mm |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Desk | control | 403 / 573 | 403 | 62.78 | 62.78 | 62.78 |
| Desk | retain | 568 / 573 | 403 | 62.78 | 63.05 | 85.79 |
| Desk | anchor | 573 / 573 | 403 | 62.78 | 65.61 | 69.82 |
| Desk | multiview | 573 / 573 | 403 | 62.78 | 64.16 | 59.02 |
| RPY | control | 679 / 694 | 675 | 77.99 | 77.99 | 77.99 |
| RPY | retain | 694 / 694 | 675 | 77.99 | 59.24 | 109.77 |
| RPY | anchor | 684 / 694 | 665 | 76.46 | 158.90 | 157.73 |
| RPY | multiview | 689 / 694 | 675 | 77.99 | 68.42 | 68.37 |
| XYZ | control | 792 / 792 | 789 | 18.49 | 18.49 | 18.49 |
| XYZ | retain | 792 / 792 | 789 | 18.49 | 19.57 | 19.57 |
| XYZ | anchor | 792 / 792 | 789 | 18.49 | 20.98 | 20.98 |
| XYZ | multiview | 792 / 792 | 789 | 18.49 | 19.64 | 19.64 |
| ScanNet | control | 160 / 160 | 160 | 60.52 | 60.52 | 60.52 |
| ScanNet | retain | 160 / 160 | 160 | 60.52 | 68.79 | 68.79 |
| ScanNet | anchor | 160 / 160 | 160 | 60.52 | 69.11 | 69.11 |
| ScanNet | multiview | 160 / 160 | 160 | 60.52 | 71.49 | 71.49 |
| ETH Room | control | 140 / 263 | 139 | 7.73 | 7.73 | 7.73 |
| ETH Room | retain | 140 / 263 | 139 | 7.73 | 10.52 | 10.52 |
| ETH Room | anchor | 140 / 263 | 139 | 7.73 | 12.92 | 12.92 |
| ETH Room | multiview | 140 / 263 | 139 | 7.73 | 6.96 | 6.96 |
| ETH Sofa | control | 357 / 533 | 325 | 35.06 | 35.06 | 35.06 |
| ETH Sofa | retain | 357 / 533 | 325 | 35.06 | 21.93 | 21.93 |
| ETH Sofa | anchor | 357 / 533 | 325 | 35.06 | 22.29 | 22.29 |
| ETH Sofa | multiview | 357 / 533 | 325 | 35.06 | 22.06 | 22.06 |

RPY multiview improves translation by 12.3 percent on all 675 scored reference
frames, adds ten connected frames and also improves whole-root error. Rotation
RMSE on that same population changes from 4.10 to 4.36 degrees. The translation
and map gains therefore do not imply uniformly better pose estimates. Using the
same full-run alignment, the original 325:345 weak interval improves from
230.25 to 67.87 mm on all 20 matched frames. Several other intervals regress.
Retention alone has an even lower matched translation error, 59.24 mm, with
109.77 mm on its larger full connected population. Its extra coverage is not
geometrically reliable enough to pass the whole-root gate. Anchor-only RPY
regresses on both populations and loses part of the reference intersection.

Desk multiview connects all 573 frames. Its 59.02 mm whole-root result and the
62.78 mm reference use different populations. On the same 403 frames translation
worsens slightly to 64.16 mm, while rotation improves from 3.82 to 3.37 degrees.
This explains why the whole-root score alone cannot justify promotion. The
matched reconstruction evidence below exposes a larger regression.

The strongest transfer wins are Sofa and ETH Room. Sofa gains on the same 325
scored frames for all three policies. Retention alone is marginally best, so the
Sofa result does not establish that triangulation caused the gain. ETH Room
multiview improves from 7.73 to 6.96 mm on all 139 scored root frames, while
retention and anchor-only regress. ScanNet regresses with unchanged connected
coverage, including an 18 percent multiview translation increase. XYZ multiview
also exceeds the predefined five percent transfer regression limit.

Tracking failures remain zero on TUM and ScanNet, one on Room and 26 on Sofa
for the main shared policies. Historical connected coverage and immediate
tracking continuity are different measurements. No main policy meets the promotion criteria.

## Observability and frozen controls

The full fresh undamped global audit fixes raw component gauges and unused
cameras. These counts apply to the attempted observation objective. They do not
certify every output attachment or the validity of the sensor model.

| Recording / policy | Reduced camera dimension | Extra numerical null directions | RPY 325:345 rigid-cut rank |
| --- | ---: | ---: | ---: |
| Desk / control | 666 | 11 | n/a |
| Desk / retain | 678 | 6 | n/a |
| Desk / anchor | 684 | 9 | n/a |
| Desk / multiview | 684 | 8 | n/a |
| RPY / control | 792 | 19 | 3 |
| RPY / retain | 828 | 5 | 6 |
| RPY / anchor | unavailable | unavailable | unavailable |
| RPY / multiview | 822 | 2 | 6 |

The original RPY one-landmark cut has rank three. Retention raises it to six
with 26 boundary landmarks. Multiview has rank six with 31 boundary landmarks.
Both improve this particular cut. At the exact saved multiview camera and point
states, a frozen ablation that excludes only initialized pure RGB classes raises
the global null-direction count from two to six. The cut itself remains rank
six. Thus pure RGB classes add collective information beyond the retention
budget and native-anchored rays in this linearized state. The ablation neither
reruns tracking nor establishes the causal accuracy attributable to those four
removed null directions.

The two remaining RPY null directions concentrate at camera 585. It retains 184
rows, including 52 native samples, but just two observations belong to initialized
multiview classes. One supplies a native sample whose other observer has only a
bearing. The other is a bearing attached to an earlier native anchor. The single
camera cut has rank four. Recent anchor checks have at most two metric
correspondences and cannot supply a verified bridge. The remaining problem is
shared geometric support despite a sizable per-view depth count.

Full RPY multiview retains 31,365 rows, including 9,010 without frontend 3D
owners. Of those unowned rows, 1,262 are eligible in the final metric objective.
There are 3,947 eligible image-only rows, including 1,535 observations in 389
pure RGB classes. The other eligible image-only rows belong to native-anchored
classes. The final partition has 165 active bearing proposal edges. Counts of
retained and eligible observations are separate from convergence or point truth.

A frozen admission audit uses the previous RPY exposure and native covariance.
It reproduces retained native bindings without reference poses. For 335 to 340,
the legacy retained target rows supply ten anchor correspondences and fail the
minimum disjoint support gate. Spatial extras supply 16 and pass. Prioritizing
native descriptor matches supplies 21 at the same target budget and also passes.
The proposed covisibility priority was audited retrospectively. It is not a
new causal runtime policy. An extra 19 candidate anchors on 325 to 340 still fail
withheld verification, illustrating that raw count is insufficient.

Controlled mixed RGB/native systems agree with a dense joint reference solve.
At identical frozen RPY multiview factors and initial states, 32-iteration full
and gauge-transported solves reach Cauchy costs 6905.736675 and 6905.736695.
Maximum pose-entry difference is 2.08e-6 and maximum point-entry difference is
4.62e-5 m. The corresponding ScanNet replay is bitwise equal. These are numerical
controls on fixed objectives. Objective values of different live partitions or
activation sets cannot be compared as external accuracy.

A separate Desk cut isolates cameras 160 and 165. Their one boundary class is
observed through an RGB ray at 165 and native depth at 175. The collective cut
has rank two, leaving four rigid directions unobserved. A finite control moves
both cameras and their internal landmark by 10 cm along the boundary ray while
holding the outside point fixed. Internal relative coordinates are unchanged,
and the boundary projection is unchanged because its camera coordinates are
scaled along the same ray. The saved objective cost is identical at reported
precision, with maximum residual change 1.14e-13. Yet the two-camera external
RMSE ranges from 187.71 to 325.58 mm across the opposite translations, versus
246.21 mm at the saved estimate. Native map samples from the moved cameras also
move by 10 cm. This verifies a finite geometric ambiguity in the actual
retained objective without mutating any saved reconstruction.

A frozen axial-depth ablation holds the input population and initialization
fixed while using only the native z marginal. On 137 RPY root keyframes,
translation RMSE changes from 68.22 to 66.10 mm. On 115 Desk keyframes it
changes from 59.53 to 59.66 mm. The two residual models have different
likelihood dimensions. These sensitivity results neither isolate exposure
error nor justify an online sensor-model change.

### RPY anchor-only rejection

The last six anchor-only solves, from received frame 665 through 690, reject
invalid projections. At the saved final state, one of 8,059 attempted rows
projects behind its color camera, with z = -0.114 m at frame 550. The frozen
graph audit reports unavailable rank for that objective and does not drop the
row to obtain a rank value.

A legacy historical association at 555:665 joins five earlier image-only
observations to a class anchored by native depth at frame 660. Its loop transform
is about 11 mm and 0.22 degrees from the depth-pose reference. Placing the native
point at the reference pose gives positive range in the earlier cameras, with
frozen-exposure RGB discrepancies between 3.85 and 10.61 pixels. This is evidence
of an incompatible initialization under the current camera and point states.
The diagnostics cannot distinguish a wrong identity, exposure mismatch and
previous camera-placement error with certainty.

A controlled frozen replay reconstructs the exact saved live partition and then
revokes only that proposal. The unchanged replay rejects. The revoked replay
refines 8,048 eligible rows to a finite objective. All raw rows remain owned and
unchanged. A changed partition changes objective eligibility, so its costs and
camera population are not directly comparable with the invalid original solve.
This experiment does not establish that online revocation would have produced
better subsequent tracking. It motivates staging historical aliases and checking
all affected class projections before committing their activation.

## Landmark consistency and causal revisions

Matching physical native samples by frame, feature, native sample and actual
native coordinate gives the following reference-placed point discrepancies.
The same sample population is scored on both sides. It is still a sensor-based
proxy rather than independent point ground truth.

| Recording | Matched native samples | Complete RMSE, mm | Multiview RMSE, mm |
| --- | ---: | ---: | ---: |
| Desk | 3182 | 48.87 | 33.70 |
| RPY | 2883 | 51.69 | 53.44 |
| XYZ | 8319 | 15.95 | 16.46 |
| ScanNet | 1284 | 31.99 | 36.39 |
| ETH Room | 1277 | 12.00 | 10.85 |
| ETH Sofa | 1517 | 9.29 | 8.11 |

RPY's matched native point discrepancy and in-sample native residual both worsen
slightly while trajectory translation and surface scores improve. Desk's native
point discrepancy improves even as its map worsens. The landmark sample proxy
covers a different physical population from the held-out surface renderings.
This is why trajectory, landmark and reconstruction results are reported
independently.

Real 120-frame Desk and 240-frame RPY prefix runs reproduce every emitted
pose, component, version and the entire immutable raw packet prefix, including
descriptors, for anchor and multiview. The 120-frame default XYZ run exactly
reproduces the protected complete reference prefix. Comparison is bitwise.

Later RPY history changes the first 235 common root poses by 14.54 mm RMS for
anchor and 2.57 mm for multiview. The corresponding prefix external errors
change from 65.26 to 69.61 mm and 70.84 to 71.76 mm. The final full RPY multiview
improvement is therefore not an improvement of the whole early prefix. Desk
anchor and multiview prefix revisions are 2.69 and 2.29 mm RMS on 120 frames.
These are retrospective revision magnitudes with separately verified exact
causal publication.

## Reconstruction

The map score is the mean held-out native target depth error capped at 20 cm,
with missing predictions charged the same cap. The target set is fixed across
policies and does not enter fusion. Its images can have entered tracking. Reverse
support projects unique reconstructed voxels into held-out native depth at
reference poses and reports support and unsupported free-space evidence.
Neither diagnostic is independent physical surface truth.

| Recording | Fusion stride | Reference full, mm | Multiview full, mm | Reference common training, mm | Multiview common training, mm |
| --- | ---: | ---: | ---: | ---: | ---: |
| Desk | 5 | 55.78 | 81.67 | 55.78 | 78.60 |
| RPY | 5 | 92.88 | 76.94 | 92.88 | 76.93 |
| XYZ | 10 | 31.66 | 34.24 | 31.66 | 34.24 |
| ScanNet | 10 | 54.14 | 56.88 | 54.14 | 56.88 |
| ETH Room | 10 | 117.30 | 116.88 | 117.30 | 116.88 |
| ETH Sofa | 10 | 26.88 | 23.33 | 26.88 | 23.33 |

RPY improves by 17.2 percent on the same 135 fusion views. Common-training
reverse supported-voxel fraction increases from 0.7817 to 0.8270 and unsupported
free space falls from 0.1577 to 0.1216. This passes the RPY map gates. Frame 445
has no training reference at stride five and is removed from all map-training
policies, with its original SLAM input preserved. This result accompanies the
matched trajectory gain, rather than being inferred from connected coverage.

Desk regresses from 55.78 to 78.60 mm on the same 81 fusion views. Unsupported
free space increases from 0.0513 to 0.0731. The largest held-out score increases
are around frames 145 through 335. On those target intervals, 145:175 and
175:335 capped errors change from 89.92 to 149.04 mm and from 68.06 to 120.32 mm.
On the 20 matched trajectory frames in 145:175, translation worsens from
86.53 to 184.10 mm. Other scored Desk intervals improve. This regional effect
and the map regression persist with fixed populations. Neither full Desk geometry
nor its matched reconstruction meets the expected preservation of prior gains.

On the four-way matched RPY fusion cohort, anchor-only improves the forward
score from 92.93 to 72.28 mm despite substantial pose regression and late solve
rejections. This is another reason to require several independent diagnostics.

All audited saved chunks reproduce exactly from original native depth at the
published poses, including component and version metadata. Exact refusion and
external reconstruction error measure different properties.

## External dense odometry control

The isolated reference uses OpenCV contrib 4.14.0.94, NumPy 2.5.4 and Pillow
13.0.0. The synthetic 40 mm translation convention check passes within 3 mm.
The six received source fingerprints exactly match the complete shared control.
An export metadata omission was corrected from the saved trajectory archive,
without altering any poses, components or reconstruction arrays. Original and
corrected report hashes are linked by export receipts. The driver now writes
complete format-2 metadata directly.

| Recording | External connected | Matched scored frames | Shared reference matched RMSE, mm | External matched RMSE, mm |
| --- | ---: | ---: | ---: | ---: |
| Desk | 158 | 158 | 64.79 | 57.35 |
| RPY | 694 | 675 | 77.99 | 182.44 |
| XYZ | 792 | 789 | 18.49 | 94.19 |
| ScanNet | 142 | 142 | 57.65 | 562.36 |
| ETH Room | 17 | 16 | 4.64 | 18.81 |
| ETH Sofa | 69 | 65 | 37.98 | 26.80 |

This control exposes the effect of fragmentation and the limits of adjacent
photometric alignment. It connects fewer frames on Desk, Room, Sofa and ScanNet.
Its apparent matched Desk and Sofa trajectory advantages concern short common
populations and cannot establish a whole-recording advantage. On RPY and XYZ it
has full connected coverage but substantially worse translation. These results
are scoped to this odometry algorithm and its recorded calibration settings.
They do not settle whether another complete SLAM system would suit SceneRecall.
The verified external map artifacts are included in the reproducible evidence.

## Computational behavior

Retained raw storage grows with admitted cameras and observations. Descriptor
archives add explicit linear storage, and pending classes also increase identity
partition and selection work. The matrix-free solver temporarily eliminates
initialized point variables. The original camera cap remains 512, and no
unbounded-memory or real-time result is established.

| Recording / policy | Pipeline seconds | Frame latency p95, seconds | Peak RSS, MiB | Retained raw rows |
| --- | ---: | ---: | ---: | ---: |
| Desk / control | 246.5 | 2.45 | 260.1 | 12615 |
| Desk / multiview | 383.2 | 4.18 | 298.0 | 27754 |
| RPY / control | 408.8 | 3.84 | 294.6 | 12381 |
| RPY / multiview | 395.5 | 3.68 | 345.1 | 31365 |
| XYZ / control | 475.8 | 3.67 | 313.5 | 19999 |
| XYZ / multiview | 465.2 | 3.73 | 393.7 | 40442 |
| ScanNet / control | 51.7 | 1.37 | 227.1 | 4096 |
| ScanNet / multiview | 48.0 | 1.33 | 237.4 | 8192 |
| ETH Room / control | 49.5 | 0.95 | 196.8 | 4212 |
| ETH Room / multiview | 99.0 | 2.10 | 207.5 | 11276 |
| ETH Sofa / control | 99.6 | 0.64 | 223.7 | 3273 |
| ETH Sofa / multiview | 171.7 | 1.55 | 234.4 | 6092 |

Runs share machine resources. Their wall times and peak RSS describe observed
pipelines, with no isolated speed comparison. A late rollback can reduce a run's
work while degrading geometry. The table and saved logs keep timing separate
from numerical convergence and scientific accuracy. Memory figures are peaks,
not fitted memory growth rates. Long-horizon scaling and budgets beyond the
current camera cap still require separate experiments.

The same RPY multiview policy grows from 11,427 raw rows at 240 frames to 31,365
at 694 frames. Their raw packet and descriptor arrays occupy 8.89 and 24.41 MiB,
including 5.58 and 15.31 MiB of descriptors. Peak process RSS grows from 206.9
to 345.1 MiB. Median graph solve time over the last ten full-run updates is
3.58 seconds, compared with 0.09 seconds for the first ten. This is a concrete
scaling limit even when whole-pipeline elapsed time compares favorably to one
previous reference run.

A frozen damped step on the final RPY multiview system has 139 cameras, 9,526
active rows and 2,719 landmarks. Across four alternating repeats, coarse
preconditioning reduces checked PCG iterations from 131 to 56, with maximum
step-entry difference 4.56e-12. Median step times are 0.211 and 0.127 seconds
under concurrent workloads. The result concerns one fixed linear system and
supports an efficiency direction without establishing causal accuracy.

## A geometric proxy for persistent scene memory

`evaluate_revisit_placement.py` compares the placement of the same native
surface samples at distant visits. Reference depth poses assign fixed 10 cm
spatial patches. Each patch needs three views and 20 samples in both the first
and last thirds of the recording, on matched root views. The metric compares
the change in mean placement error between those visits, using the same native
samples under both policies. It is an additional diagnostic. It does not use
semantic identities or assert that all points in a patch are one physical surface.

| Recording | Shared patches | Complete p95 shift, mm | Multiview p95 shift, mm | Complete within 5 cm | Multiview within 5 cm |
| --- | ---: | ---: | ---: | ---: | ---: |
| Desk | 363 | 63.21 | 63.66 | 0.835 | 0.931 |
| RPY | 557 | 62.41 | 54.97 | 0.876 | 0.910 |
| XYZ | 411 | 15.23 | 15.28 | 1.000 | 1.000 |
| ScanNet | 167 | 47.75 | 53.78 | 0.982 | 0.862 |
| ETH Room | 0 | unavailable | unavailable | unavailable | unavailable |
| ETH Sofa | 334 | 16.08 | 15.66 | 1.000 | 1.000 |

The Desk patch improvement coexists with worse complete surface reconstruction.
These patches measure repeat placement in spatial regions seen in both visits.
They do not cover all the surfaces a memory query may need. Missing late root
coverage or lack of shared patches with sufficient observations makes some
recordings unscorable on this proxy. No received SLAM input is removed.

A provisional target for the next scene-memory trial is at least 95 percent of
matched revisited patches within the existing 5 cm consistency tolerance, with
no loss of connected reference coverage. This target needs calibration against
annotated object revisits and retrieval accuracy. For a live trial, use the
input frame period as the active tracking latency budget and one second as a
provisional revision publication budget. The current serial pipeline does not
meet those budgets. Freeze application-specific targets before further trials.


## Desk compatibility with support-aware retention

Two complete Desk combinations hold the earlier support-aware policy as their
reference. The first multiview attempt was interrupted before export. It was
restarted from frame zero with the same frozen input range and policy. The
interrupted log and restart record are retained.

| Policy | Connected / 573 | Matched scored frames | Reference matched RMSE, mm | Candidate matched RMSE, mm | Whole-root RMSE, mm |
| --- | ---: | ---: | ---: | ---: | ---: |
| Support alone | 568 | 568 | 73.05 | 73.05 | 73.05 |
| Support plus retain | 573 | 568 | 73.05 | 83.86 | 89.64 |
| Support plus multiview | 573 | 568 | 73.05 | 40.92 | 41.20 |

On the 399 frames shared with both the original complete and support-only
references, the three scores are 61.67, 43.95 and 40.12 mm, respectively.
The combination preserves and extends the matched trajectory gain. Its fresh
undamped graph has zero detected extra null directions in 1,416 reduced camera
dimensions. In the same received interval 160:170, a denser set of cameras has
59 boundary classes and a rank-six collective cut. The original multiview cut
with the ordinary cadence has only one boundary class and rank two. The camera
sets differ, so this is an interval comparison rather than an identical-variable
linearization. Against the original complete reference, 372 matched native
samples also improve in reference-placed point discrepancy from 44.54 to
29.97 mm. The support-only reference has no final optimized point cache after
its rejected last solve, so a point-cache comparison against it is unavailable.

Reconstruction remains mixed. On the same 114 fusion views, the combined
multiview forward score is 54.31 mm versus 45.03 for support alone. Target
coverage changes only from 0.9734 to 0.9721. Reverse support improves from
0.8717 to 0.9327, and unsupported free space falls from 0.0430 to 0.0285. On the
80 fusion views also shared with the original complete reference, the scores
are 56.02 for complete, 46.22 for support alone and 50.52 for the combination.
Retention without multiview worsens the matched support-only forward score to
66.01 mm. The combined multiview policy does not preserve the earlier forward
map gain within the proposed tolerance. It was tested only on Desk.

The full Desk combination takes 1,399.5 seconds and peaks at 400.3 MiB with
55,580 retained raw rows. The prior support-only run takes 954.4 seconds and
peaks at 273.4 MiB with 21,363 rows. These are observed pipeline timings from
separate runs. The gains do not establish a scalable live configuration.

## Reproducibility and decision

The complete reference and validated defaults remain unchanged. Multiview
passes the predefined RPY trajectory and reconstruction gates. ScanNet and XYZ
exceed the transfer regression limit. The Desk combination strengthens the
trajectory case but leaves mixed reconstruction and remains untested on other
recordings. No policy is promoted.

The most direct next step is staged activation of historical landmark aliases.
Verify affected classes against all their retained rays, and try geometrically
supported reinitialization before committing a merge. Use collective boundary
information to distinguish a weakly attached group from a usable metric join.
The RPY anchor rollback and Desk finite ambiguity provide concrete regression
cases. Exposure transforms remain frozen estimates, including the nominal
initial exposure. Joint timing and calibration states remain a separate research
step. Efficient incremental factorization becomes more valuable once those
association and geometry contracts are stable.

Evidence is under ignored `outputs/bearing_shared`. `full_manifest.json` records
all six cases and the three main policies. The study driver accepts that file
with a fresh output root. `promotion_protocol.json` and `promotion_assessment.json`
record the frozen gates and primary outcomes. The matched comparisons, frozen
replays, cut audits, revisit diagnostics and exact map reports retain their
source hashes and scopes. Per-worker protocols and source snapshots identify
the code used for each run. External dependencies are isolated and pinned in
`reference_requirements.txt`. They do not modify the project lock.

Post-study validation tightens raw provenance, rejects mismatched owned pixels
before linking, and preserves the prior positional configuration interface.
A fresh full RPY multiview run with those checks reproduces all 94 arrays in
the six state archives bitwise. Fresh default XYZ and anchor Desk prefixes
also reproduce their saved emitted and immutable raw prefixes bitwise.
The corrected independent OpenCV exporter passes a 120-frame format check.

Final checks pass 1,141 tests, including 25 new bearing tests, plus Ruff and
whitespace checks. All 967 protected artifact paths match their original hashes.
The integrity report verifies 36 new raw runs, three package snapshots, three
study/reference driver snapshots and 170 exactly rebuilt reconstruction chunks.
Raw feature and native ownership, as well as one observation per live camera
class, hold in every checked run. No commit was created.
