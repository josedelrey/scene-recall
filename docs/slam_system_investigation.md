# SLAM system diagnosis and research directions

This investigation extends the [incremental SLAM](incremental_slam.md) and
[landmark lifecycle](slam_landmark_lifecycle.md) studies. Sparse and the original
SLAM defaults remain unchanged. The new implementations are offline experiments.
They preserve the existing estimators, measurement archives and evidence capsules.

The central finding is that SceneRecall has several different error regimes.
Insufficient overlap can prevent optimization altogether. Weak geometry and a
small measurement budget can make a valid optimization statistically worse than
its initialization, even with a correct measurement model. Physical-model error
and uncertain identities add further risks. Neither accepting more updates nor
lowering their cost identifies which regime applies.

## Questions and experimental boundaries

The source-free study reuses native SLAM factor packets from all fifteen earlier
populations, with the corresponding no-optimization trajectory as initialization.
The packets fix identities, pixel coordinates, native depth samples, exposure
transforms and whitening. Initial landmarks are rebuilt from their first valid
depth sample **inside each block**. Historical final landmark coordinates do not
initialize the experiment. References enter scoring only.

Each segment is partitioned into disjoint complete eight-camera blocks. Tracks
need at least two observations and a local native-depth seed. Incomplete blocks,
unsupported splits and missing metric factors are reported without replacement.
Track IDs modulo four define four deterministic folds. An entire fold of landmark
identities stays outside pose and training-point optimization. A test landmark
is initialized from its first withheld depth observation, then predicts later
withheld RGB and depth measurements. That seed observation is excluded from scoring.

The primary held-out summaries average within track, then within block. RGB
errors are capped at 10 pixels and axial depth errors at 0.1 m. Raw errors, counts
and invalid projections remain in the reports. These scores include seed noise.
They test cross-view prediction, rather than independent physical surface truth.

The four fixed measurement models are:

| Model | Controlled change |
| --- | --- |
| Native | Existing RGB and three-dimensional native-depth objective |
| Axial | Retain RGB and only native z with its marginal variance |
| Weak depth | Multiply native depth standard deviation by four |
| First depth | Retain the earliest valid native depth per training track and every RGB observation |

All use the same bounded Schur batch solver, Cauchy loss, 20-iteration limit,
0.1 m camera and point displacement limits, and 5 degree rotation limit. This
removes the marginal prior while leaving association and measurement assumptions
fixed. It is a diagnostic replay, without an online accuracy claim.

The split occurs after frontend association and native graph admission. A withheld
identity could already have influenced the frontend trajectory or other identities.
Nearby tracks share calibration and scene errors. Accordingly, this is a stronger
optimizer check than in-sample residuals, but it is neither end-to-end measurement
isolation nor an independent-scene validation. Folds and windows from the same
recording are repeated measurements of that recording, not independent scenes.

## Findings

The archived fold-zero population contains 92 eligible blocks across thirteen
metric populations. Both ScanNet populations have no admitted metric factors.
Four-fold replay produces 367 block/fold evaluations. One block lacks a supported
split in fold three. These folds overlap in their training identities and are
not 367 independent trials.

### Objective fit, prediction and external accuracy differ

Removing marginalization does not remove the mixed accuracy rankings. For example,
in fold zero, late FR2 Desk's mean capped withheld RGB error falls from 2.46 to
1.05 pixels under Native replay. Its mean block-anchored translation RMSE changes
from about 10 to 12 mm. Late Office changes from about 6 to 8 mm while prediction
improves from 1.25 to 1.19 pixels. Full XYZ's sampled blocks improve withheld RGB
prediction from 2.15 to 1.61 pixels while mean anchored rotation RMSE rises from
1.02 to 1.24 degrees. These are local, disjoint block metrics, distinct from
whole-sequence aligned ATE.

A validation score built from the same imperfect RGB-D geometry can therefore
favor a change that disagrees more with the external trajectory. This evidence
rules out a universal explanation based only on frozen marginalization or fitting
the training landmarks. It does not identify calibration, timing, reference error
or false association as the unique physical cause of a particular real-data error.

Across the four folds, Native refines 65, 66, 66 and 66 blocks. Four refined blocks
per fold lack enough reference poses for external comparison. Withheld RGB
prediction improves while position RMSE worsens in 19, 19, 20 and 20 blocks.
This conclusion survives the split change. It is not a calibrated probability
that a future update will fail.

### Linear-solve roundoff is small in the controlled systems

The numerical diagnostic constructs the same robustly weighted, column-scaled
Jacobian for square-root least squares and normal-equation solving. Both
undamped and identically damped steps are checked on fifteen generated near,
far and planar systems. The maximum relative step disagreement is 1.17e-12,
and the maximum absolute camera-coordinate disagreement is 1.01e-13.

These float64 comparisons give no evidence that linear-solve roundoff explains
the millimeter-scale errors in these controls. They do not exclude nonlinear
linearization, marginal-prior inconsistency, accumulated conditioning effects or
float32 problems. The prior studies' square-root retirement tests remain relevant.

### A correct model can still worsen accurate input poses

Controlled scenes contain eight cameras, known identities and exact calibration.
RGB noise is independent, one-pixel Gaussian. Native XYZ noise has known diagonal
standard deviations: z/525 laterally and 0.005 + 0.003 z² meters axially. There
are no historical priors, incorrect exposure transforms or association mistakes
in the independent-noise control. Seeds 0 through 4 are fixed. Near, far and planar
geometry are evaluated under mixed translation/rotation and predominantly
rotational motion.

A separate information-budget experiment uses the exact independent Gaussian
likelihood, wider inactive update bounds and 40 iterations. It varies available
tracks while pairing the same input pose perturbations across budgets and
interventions. Three quarters of the tracks train the solver. The table gives
mean root-anchored translation RMSE over five seeds for mixed motion, in mm.

| Geometry | Accurate initialization | 72 training tracks | 288 training tracks | 1,152 training tracks |
| --- | ---: | ---: | ---: | ---: |
| Near, volumetric | 7.22 | 3.12 | 1.64 | 0.84 |
| Far, volumetric | 7.22 | 6.96 | 3.46 | 1.79 |
| Plane | 7.22 | 23.34 | 14.75 | 6.67 |

Starting from noisier, 36.12 mm input poses gives essentially the same fitted
results. This isolates the information in the objective from initialization
quality. A small, geometrically weak bundle can discard information implicit in
a better frontend estimate. Full rank alone does not guarantee a useful update.
The tested planar scene is observable, with substantially greater pose variance.
Increasing correct observations reduces that variance, with additional work.
These constructed scenes establish a mechanism, not the fraction of real errors
attributable to it or a calibrated posterior for SceneRecall.

The implication is to retain more of the frontend's verified measurement evidence
and its correlations, or use an uncertainty-consistent sufficient representation.
Adding a pose prior from the same RGB-D samples without managing ownership would
count evidence twice. A tuned displacement threshold cannot substitute for that
information contract.

### Sensor and identity interventions have different effects

The paired controls separately introduce a frame-correlated sinusoidal depth
bias with 25 mm amplitude, a 2% focal error, omitted exposure motion, or persistent
RGB/depth identity swaps on about 10% of tracks in the latter half of the window.
Every intervention preserves the initial poses and physical reference trajectory.

For mixed-motion near geometry, Native's mean position RMSE changes from 3.19 mm
under independent noise to 7.71 mm under correlated depth bias. Weak depth changes
from 4.90 to 5.03 mm. In the independent planar case, Native is 25.33 mm and Weak
depth is 37.55 mm. Suppressing depth can help a particular corrupted regime while
losing useful information elsewhere. First-depth factors remove repeated depth
information rather than estimating or correcting the bias.

The deliberately large identity swaps are mostly suppressed by the existing
robust objective in this construction. That is a positive control for this
outlier pattern, without evidence against subtle natural mismatches. A persistent
small wrong match, a moving rigid object, or a calibration deformation can remain
internally consistent. Reference-relative residuals alone cannot label identities.

TUM's registered-image adapter intentionally uses the documented common ROS
pinhole approximation. Replacing just the RGB intrinsic matrix with another
published calibration would leave depth rays, registration and uncertainty
inconsistent. Full physical calibration needs matched image/depth models and
independent capture-level validation. The official [TUM format guidance](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)
also warns about undistorting registered depth. [BAD SLAM's camera requirements](https://github.com/ETH3D/badslam/blob/master/applications/badslam/doc/camera.md)
explain synchronization and depth-deformation requirements relevant to this risk.

### Observability needs explicit submap gauges

The component experiment partitions the training camera/landmark graph. Each
component fixes its first input camera and refines only its internally observable
relative geometry. Single-camera components and geometrically deficient solves
remain unchanged. This preserves separate gauges and supplies no information
about the transform between components.

In fold zero, more Desk, RPY and late Room blocks can refine under this policy.
Across all fold-zero blocks, Native refinement availability changes from 65/92
to 75/92. Desk changes from 1/9 to 5/9, RPY from 1/6 to 4/6, late Room from 1/6
to 3/6, and early Room remains 0/4.
Prediction often improves, while external pose accuracy remains mixed. Restoring
local update availability therefore does not establish a globally valid submap
connection. The implementation is useful as a controlled prototype for explicit
gauge handling. It is not a replacement for the coupled online marginal prior.
These are withheld-training graph failures, rather than replays of the exact
online graph with its historical prior.

Some connected graphs still fail the undamped condition gate. Combinatorial
connectivity is necessary but insufficient. Rank, geometric leverage, uncertainty,
overlap and dynamic-scene consistency need separate checks. Disconnected graphs
cannot be repaired by damping or by assigning artificial certainty to a new root.

### Full-rate output is another approximation

The current archive attaches nonkeyframes rigidly to their preceding keyframe.
The audit groups one-frame RPE by the five-frame optimization cadence. Early
FR2 Desk's translation RMSE at phase zero is 8.32 mm for Native and 3.28 mm for
its matched control. Late Desk is 5.55 versus 2.30 mm at that phase. Other phases
are much closer. This pattern is consistent with revision and attachment effects,
without proving that all boundary error comes from interpolation.

An offline alternative preserves every final keyframe and segment boundary, then
interpolates camera centers linearly and rotations along the shortest arc using
timestamps. It requires the following keyframe and assumes smooth interkeyframe
motion. Trailing poses remain unchanged. It does not improve keyframe estimation.

Across the old fifteen populations it often reduces position ATE, with mixed
orientation and local-motion results. For example, Structure late changes from
15.90 to 12.42 mm ATE while anchored rotation RMSE changes from 0.661 to 0.695
degrees. Full Desk changes from 59.47 to 59.66 mm. This is an output-model
experiment, not evidence for a universally accurate motion prior.

The original dense-fusion cadence selects every tenth frame. Those frames are
SLAM keyframes, so that protocol cannot reveal intermediate-pose changes on
regular-cadence segments. A separate frozen mapping diagnostic uses stride four,
offset two and at most sixty training views per chunk. Native and interpolated
trajectories use identical inputs, target views, voxel size and fusion settings.
It must remain separate from the original reconstruction rankings.

## New capture validation

Two official TUM recordings were downloaded after freezing windows 0:300 and
400:700, existing Sparse/control/Native settings, and reconstruction rules.
Pioneer 360 exercises a robot turn. Walking Static exercises moving people with
a mostly stationary camera. They add 1,200 distinct inputs, bringing the retained
old/new accuracy populations to 6,459 distinct inputs. They are new recordings
and motion conditions within established TUM sensor families. They do not supply
a new manufacturer, an independent building, or a second ScanNet capture.
The official [TUM sequence descriptions](https://webshare.cvg.cit.tum.de/g/rgbd/dataset/)
describe their acquisition conditions.

The original archives contain duplicated reference timestamps. The existing
preparation tool omitted all two ambiguous rows in Walking Static and all six
in Pioneer, in separate hard-linked copies. Original images and tables are
unchanged. The prepared canonical recordings have 717 and 830 pairs. No pose
was selected from conflicting rows and no reference was extrapolated.

The following numbers are aligned position ATE in mm. For fragmented Pioneer
runs, ATE scores only the largest segment. Loss counts and segment sizes must
accompany those values.

| Window | Sparse ATE, losses, largest segment | Control ATE, losses, largest segment | Native ATE, losses, largest segment |
| --- | --- | --- | --- |
| Walking 0:300 | 371.9, 0, 300 | 53.3, 0, 300 | 55.1, 0, 300 |
| Walking 400:700 | 593.1, 0, 300 | 75.9, 0, 300 | 73.5, 0, 300 |
| Pioneer 0:300 | 7.9, 124, 33 | 16.7, 113, 69 | 16.7, 113, 69 |
| Pioneer 400:700 | 46.9, 70, 70 | 60.3, 41, 70 | 50.7, 41, 70 |

Each window has 299 possible transitions. Sparse's apparently attractive Pioneer
ATE comes from short, different segments, so these values do not establish its
accuracy advantage. Native accepts only five and four joint updates on the two
Pioneer windows and retains all control failure boundaries. Local BA cannot
repair the missing temporal associations or join those disconnected trajectories.
Global recovery and tracking continuity are therefore significant engineering
priorities, beyond reducing local objective values.

Walking's improvement against Sparse is principally frontend evidence. Joint
optimization worsens early ATE slightly and improves late ATE slightly. Late
anchored rotation RMSE changes from 2.51 to 2.74 degrees. Its 38 and 47 accepted
updates do not establish uniformly better motion estimation in a dynamic scene.

The frozen fold-zero replay adds eighteen eligible blocks. Early Walking's Native
mean block-anchored position RMSE changes from 29.92 to 18.40 mm, while late
Walking changes from 34.65 to 35.89 mm. Late Pioneer's two sampled blocks change
from 98.74 to 129.45 mm. Those sampled segments differ from its largest-segment
ATE population. Component refinement adds one late-Walking update, with mean
position RMSE of 36.20 mm. No model or component policy establishes a consistent
gain on these fresh recordings.

The already specified interpolation rule reduces Walking ATE from 55.07 to
40.61 mm and from 73.52 to 58.44 mm. One-frame translation RPE changes from
28.31 to 6.98 mm and from 33.76 to 12.33 mm. Every final keyframe is identical.
Pioneer largest-segment ATE changes from 16.69 to 15.14 mm and from 50.66 to
42.64 mm, with all failures preserved. Late Pioneer orientation worsens from
4.086 to 4.274 degrees. This is evidence that the full-rate output approximation
can matter substantially, with smooth-motion and delayed-output qualifications.

The original cadence-ten reconstruction comparisons are withheld on Pioneer
because Sparse and the local methods have different segmentation. Walking's
first TSDF population is withheld for missing training references. Late Walking
scores 23 TSDF target views. Capped reference-target depth error is Sparse
180.55 mm, Control 90.36 mm and Native 86.66 mm. Reference-pose fusion itself is
70.05 mm. Moving surfaces, visibility and sensor error remain in that control,
so this is not an independent static-surface accuracy measurement.

The separate cadence-four interpolation diagnostic retains all five planned
populations. Values are Native → Interpolated capped reference-target depth
errors in mm. Training, targets and settings are paired within each row.

| Population | Surface targets | Surface error | First-chunk TSDF targets | TSDF error |
| --- | ---: | --- | ---: | --- |
| Full XYZ | 136 | 28.24 → 28.81 | withheld | missing training references |
| Walking 0:300 | 15 | 48.59 → 49.70 | withheld | missing training references |
| Walking 400:700 | 74 | 164.94 → 165.43 | 59 | 84.75 → 84.13 |
| Pioneer 0:300 | 46 | 78.51 → 76.95 | 17 | 40.28 → 38.18 |
| Pioneer 400:700 | 63 | 115.75 → 116.02 | 17 | 149.05 → 150.77 |

There are 334 surface and 93 TSDF target views in this additional protocol.
Missing references withhold chunks without replacement, explaining why early
Walking's surface result covers only fifteen targets. Pioneer surface scores
pool local chunks across its segments, without joining their gauges. Its TSDF
uses only the first chunk. Neither represents a complete globally connected map.
Coverage and common-mask errors are retained alongside capped scores. Late
Walking surface coverage changes from 97.56% to 97.05%, despite better full-rate
trajectory scores. Interpolation is therefore a useful output-model direction,
with no consistent downstream geometry advantage established here.

## Architectural priorities

1. **Preserve information and verified overlap.** Replace unconditional temporal
   eviction and a small static point budget with covisibility-aware, spatially
   distributed observation selection. Measure predicted camera uncertainty and
   actual external error, rather than only counts and rank. Keep a replayable
   local factor store so returning identities can recover their geometry.
   Unify the evidence passed from tracking into smoothing. The current frontend
   and small joint graph optimize different populations, and frontend-only
   landmarks follow their latest revised observer approximately. Frozen-packet
   replay does not isolate that feedback from measurement selection on a live run.
2. **Represent gauges and recovery explicitly.** Maintain local submaps when
   global overlap is insufficient. Retain uncertainty between submaps instead
   of manufacturing constraints. Verified place recognition, relocalization and
   loop closure should reconnect them and revise all dependent outputs.
3. **Validate the sensor model across captures.** Separate RGB/depth exposure,
   clock offsets, registration, distortion and correlated depth deformation.
   Use stationary targets, repeat passes, independent surfaces and motion-rich
   calibration sequences. Estimate nuisance parameters only with sufficient
   excitation and evaluate on a different recording and camera family.
4. **Use a scalable solver for richer evidence.** The existing compact Schur
   batch backend demonstrates useful point-budget scaling. The online coupled
   prior can correlate landmarks, so independently eliminating its point blocks
   would be incorrect. Sparse square-root factorization, structured submaps or
   replay are credible directions. A solver substitution alone cannot correct
   association or sensor bias.
5. **Connect geometry to memory through provenance.** SceneRecall's durable
   entities should own source observations, timestamps and revision versions,
   independent of graph track IDs. Separate static structure from movable
   objects and maintain uncertain spatial relations across disconnected maps.
   Evaluate object association and spatial-query correctness after pose changes,
   alongside trajectory and reconstruction metrics.

For low texture, rapid rotation and prolonged occlusion, synchronized inertial
measurements are another useful source of constraints. They require a calibrated
camera/IMU transform, timing and bias model. Existing RGB-D-only archives cannot
test that option. Moving rigid objects can support a coherent alternative camera
motion, so robust residuals alone do not establish a static world. Scene-level
motion reasoning and semantic entities have value beyond masking a few outliers.

Relevant architectural precedents include [ORB-SLAM3's multi-map system](https://arxiv.org/abs/2007.11898),
[iSAM2's sparse incremental relinearization](https://www.cs.cmu.edu/~kaess/pub/Kaess11icra.html),
[square-root marginalization](https://arxiv.org/abs/2109.02182), and
[BundleFusion's geometry reintegration](https://arxiv.org/abs/1604.01093).
These papers motivate specific design choices, without predicting a measured
gain in SceneRecall. [DROID-SLAM](https://arxiv.org/abs/2108.10869) supplies a
learned correspondence/dense-BA comparison direction. [DynaSLAM](https://arxiv.org/abs/1806.05620)
motivates explicit dynamic-scene treatment. Neither method was installed or
benchmarked here. A learned frontend should first compete under the same sensor,
failure, mapping and compute protocol, with a fixed training/test separation.

## Remaining limitations

Remaining physical attribution requires independently checked correspondences,
surface truth and calibrated timing. This study does not estimate a natural
false-match rate, learn a transferable sensor model, implement global recovery,
or establish calibrated real-data covariance. Covisibility selection, sparse
coupled-prior optimization and conditional recovery of retired geometry remain
implementation work. Two-sided interpolation needs validation under acceleration
and lower frame rates before any online use. Native Windows and hosted CI were
not exercised in this investigation.

## Computational performance and verification

Serial raw-input timing ran after the accuracy, reconstruction and test jobs.
The same eighty-frame XYZ prefix was processed in forward and reverse backend
orders on the i7-13700HX, with one OpenBLAS/OMP/OpenCV thread and warm files.
Sparse runs at 4.40/4.39 fps, Control at 3.42/3.55 fps and Native at 3.32/3.42 fps.
Native keyframe p95 is 657/707 ms and median joint-solve time is 92/86 ms.
This is short-prefix CPU evidence, with shared-machine variability and no claim
of real-time camera-rate processing or significance of small order differences.

The separate serial Gaussian information-budget replay measures only eight-camera
batch optimization. Median solve time for 72/288/1,152 training points is
33/134/502 ms on near volumetric geometry, 37/130/498 ms on far geometry and
64/230/886 ms on planar geometry. These are different fixed optimization problems,
without raw extraction, tracking or coupled historical priors. They demonstrate
the computational cost of the richer evidence that reduced the toy pose errors.
Concurrent accuracy-run times are excluded from performance conclusions.

Linux verification passes 903 tests, Ruff lint/format and whitespace checks.
The new tests cover measurement isolation, gauge invariance, exact-geometry
stationarity, component roots, paired interventions, source/calibration identity
and interpolation boundaries. A complete raw Walking 0:300 Native run exactly
matches the cached trajectory, emitted trajectory and measurement arrays.
Protected evidence-capsule hashes and the old eighty-frame Sparse/control/Native
trajectory repeats are checked separately. No original baseline was overwritten.

## Reproduction

Ignored evidence lives under `outputs/slam_system_study/`. The pre-existing
[evidence capsules](output_storage.md) remain authoritative and unchanged.
`cases.json` maps each case to a native measurement run and a matched initial
trajectory. Archived runs can be restored into a separate directory. The new
source-free drivers need no dataset images.

Authoritative diagnostic reports are in `release/`: `joint_fold0` through
`joint_fold3`, `components`, `new_joint`, `new_components`, `synthetic`,
`information`, `numerics.json` and the two system audits. `new_sequences/` retains
the four original-policy accuracy and cadence-ten mapping comparisons.
`interpolation_verified/` and `new_interpolation/` retain revised trajectories.
The cadence-four mapping results are in `reconstruction/xyz/xyz/` and
`reconstruction/new_sequences/`. Runtime roots are `serial_raw/` and
`serial_information/`. Development and superseded runs are excluded from the
final manifest's scientific selection.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
uv run python experiments/slam_generalization.py CASES.json \
  --holdout-fold 0 --output-dir FRESH_JOINT
uv run python experiments/slam_generalization.py CASES.json \
  --solver components --output-dir FRESH_COMPONENTS
uv run python experiments/slam_model_controls.py --output-dir FRESH_CONTROLS
uv run python experiments/slam_information_budget.py --output-dir FRESH_BUDGET
uv run python experiments/diagnose_slam_numerics.py --output FRESH_NUMERICS.json
uv run python experiments/audit_slam_system.py CASES.json \
  --prediction-dir FRESH_JOINT --output FRESH_AUDIT.json
uv run python experiments/slam_pose_interpolation.py CASES.json \
  --output-dir FRESH_INTERPOLATION
uv run python experiments/evaluate_slam_interpolation.py MAP_CASES.json \
  --output-dir FRESH_MAPS
```

Repeat the joint diagnostic with folds one through three for split sensitivity.
`MAP_CASES.json` maps each case to `native` and `interpolated` trajectory directories.
Use fresh destinations. Reports preserve input and source hashes, solver status,
prediction counts, missing-reference coverage and exclusions. Concurrent runs
support accuracy comparisons, without a latency ranking. Controlled Gaussian
results are mechanisms, not replacements for independent real captures.

Production source, dependencies and lockfile are unchanged. No commits or Git
synchronization are performed by these experiments.
