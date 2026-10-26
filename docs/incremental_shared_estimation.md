# Incremental shared observation estimation

This follow-up keeps the [complete shared solver](shared_estimation.md) as its
scientific reference. It adds optional selective observation Jacobians, a
two-level Schur preconditioner, retryable numerical damping and recent-view
admission. The original solver, observation allocation and all validated
defaults remain unchanged. No dependency was added.

## Design choice and research

[iSAM2](https://people.csail.mit.edu/kaess/pub/Kaess12ijrr.pdf) makes incremental
nonlinear inference efficient through affected-clique updates, variable
reordering and selective relinearization. A Bayes tree remains a promising
future replacement for repeated global factorization. SceneRecall currently
has an iterative matrix-free solver, changing landmark identities and explicit
component gauges. A direct migration would also need factor removal and
reinsertion for identity edits, gauge changes, robust block updates and exact
custom native RGB-D factors. This study first isolates derivative reuse from
linear-solver improvements against the existing complete objective.

[Subgraph preconditioning](https://publications.ri.cmu.edu/subgraph-preconditioned-conjugate-gradients-for-large-scale-slam)
motivates improving iterative convergence without changing the measurement
model. [Visibility-based bundle preconditioning](https://homes.cs.washington.edu/~sagarwal/vbp.pdf)
uses camera-point connectivity to construct useful approximate solves. Our
implementation uses temporal camera aggregates and an exact coarse projection
of the current Schur operator. It does not implement either paper's specific
preconditioner. Aggregate ordering is deliberately simple and independently
testable.

The sensor remains an important limitation.
[Khoshelham and Oude Elberink](https://pmc.ncbi.nlm.nih.gov/articles/PMC3304120/)
measure range-dependent depth error and resolution, and explain the need for
calibration to remove systematic errors. Increasing the number of observations
under a fixed independent noise model cannot remove an unmodeled common bias.
Our axial-depth control is a residual-model ablation, without a claim of
calibrated uncertainty or improved physical sensor accuracy.

## Underlying objective

The owned nonlinear objective remains

```text
F(C, X, a) = sum_o w_o rho(||project(C_i^-1 X_a(o)) - u_o||^2)
           + sum_(o with native depth)
             w_o rho(||W_o (E_o^-1 C_i^-1 X_a(o) - d_o)||^2)
```

Each raw RGB observation and native depth sample still has one recorded owner.
The live identity partition `a` is built from reversible loop-owned unions.
There is no pose-summary likelihood, imported trajectory prior, frozen marginal
factor or independent second local objective. Fixed roots condition the measured
raw components exactly. Unmeasured cameras and nonkeyframes retain the existing
declared output attachments.

As in the reference, the joint solve includes depth-supported classes observed
in at least two recorded views. Singleton packets remain in the ledger. Their
standalone minimized terms cannot constrain a relative camera pose. Bearing-only
classes without a retained native initialization are not triangulated by this
backend, so it does not exploit every potentially useful RGB constraint.

An axial control replaces each native residual with its whitened native z
component using the existing marginal z variance. This changes the objective.
Costs from axial and native models therefore cannot be compared directly.
Admission changes also change the measurement population. Numerical comparisons
must hold both the packet population and identity partition fixed.

## Selective endpoint derivatives

`slam/shared_linearization.py` stores one camera Jacobian and one point Jacobian
per observation, with the camera and point states where they were evaluated.
For every nonlinear iteration it computes **all current exact residuals and
robust weights**. It refreshes an observation's derivatives if either endpoint
exceeds its threshold or its reuse age expires. Experimental thresholds are
2 mm camera-center translation, 0.1 degree camera rotation, 2 mm world-point
movement and eight intervening derivative evaluations.

The camera derivative uses the existing left-pose perturbation convention in
the fixed component chart. Reuse is a Jacobian approximation in that same
tangent coordinate system. It is not an approximation to the stored nonlinear
measurements. Point movement invalidates every observation of that point, even
when its camera did not move. Small changes accumulate relative to the last
derivative evaluation, rather than the previous iteration.

The cache also precomputes fixed exposure and whitening geometry and uses a
residual-only trial evaluator. The reference computes unused trial Jacobians and
revalidates reconstructed packets. Runtime differences therefore include these
kernel changes as well as selective derivative refresh. Zero-threshold tests
check agreement with the complete solver, without claiming bitwise equality for
every nonlinear stream.

Binding a new solve aligns entries by camera and live point identity and checks
pixels, native samples, exposure transforms, whitening, observation weights,
intrinsics and depth model. Removed entries are discarded. Identity partition
edits and component joins discard the whole cache. A rejected solve also
discards it. The reference's raw rebuilding and seed initialization policy is
preserved, including its conservative rebuilding of unaffected points after
identity edits.

Each iteration assembles the complete weighted normal system from the current
residuals and current or reused Jacobians. Point elimination still retains all
cross-camera terms. Every proposed step is accepted only if the exact complete
current nonlinear objective decreases within the existing movement limits.
Before stopping for a stationary gradient, small update or unsuccessful trial,
the solver requests a full derivative refresh if its last derivatives were
mixed. The fixed nonlinear iteration budget can expire before this confirmation.
The report then retains the iteration-limit reason. No convergence guarantee
or accuracy equivalence to the reference is implied.

The strict observability policy forces fresh derivatives at every iteration.
Partial-policy rank diagnostics explicitly report whether their Jacobians are
current. Conditional camera blocks never certify global observability. Neither
derivative reuse nor damping supplies missing geometric evidence.

Mixed derivative anchors can perturb the current linearized nullspace, especially
when a shared point moves below its refresh threshold. Exact objective acceptance
does not make that approximate curvature an exact information matrix. The cache
therefore carries no covariance or observability-preservation guarantee. The
fresh complete solver and offline rank audits remain necessary controls. The
study does not isolate how much causal error comes from this approximation versus
subsequent changes in tracking and admitted measurements.

## Two-level exact-system preconditioning

For the damped joint normal equations, landmark elimination gives

```text
S = B_lambda - E D_lambda^-1 E'
b = -g_C + E D_lambda^-1 g_X
S delta_C = b
```

The complete matrix-free `S` and the existing point back substitution are
unchanged. Let `Z` map one six-dimensional left-pose increment per aggregate to
the free cameras in that aggregate. Define

```text
A_coarse = Z' S Z
M^-1 = blockdiag(S)^-1 + Z A_coarse^-1 Z'
```

The coarse matrix is assembled by **summing all observation cross blocks for
each landmark within each aggregate before eliminating the landmark**. Summing
already separated diagonal contributions would lose the within-aggregate
correlations and produce a different coarse matrix. Dense-reference tests check
this identity for multiple fixed-root sets.

For SPD `S`, both the block-diagonal inverse and the additive coarse correction
are positive definite or positive semidefinite, so their sum is a valid SPD
preconditioner. Cholesky checks the coarse system. The implementation builds its
inverse once per damped trial, then uses matrix products during PCG. It groups
at least eight adjacent free cameras, increasing group size to retain at most
48 aggregates. The dense coarse dimension is at most 288, regardless of history
size. No full camera Schur matrix is materialized.

PCG still checks `||b - S delta|| / ||b||` against the original tolerance of
`1e-8`, with the existing factor-of-two true-residual check. Preconditioning
changes the numerical iteration path, not the linear system or observation
weights. Finite tolerances and nonlinear stopping can nevertheless change the
subsequent causal stream.

The optional retry policy increases LM damping after a failed checked linear
solve, within the existing ten-trial budget. Previously accepted finite
objective decreases remain usable if a later trial cannot progress. The
reference's immediate numerical-failure rollback remains its default behavior.
Algorithmic damping is never retained as a factor or a measurement.

## Evidence admission

The optional `covisibility` policy prioritizes spatially spread owners shared
with the most recent recorded camera in the same tracking segment. It allocates
up to three quarters of the observation budget to existing owners and reserves
a quarter for new native observations, filling spare capacity from either set.
Remaining historical owners follow recent owners. The original balanced policy
keeps its exact previous ordering.

Admission never duplicates native samples or features and never adds a prior
edge to conceal a disconnected raw graph. It cannot create a link if no usable
shared frontend owner survives. Preferring recent links can also remove useful
older or future evidence. This remains an evaluated experimental policy.

## Use and reproduction

The existing command can select each change independently:

```bash
uv run python -m scene_recall.slam.cli --tum-path /path/to/recording \
  --output-dir outputs/my_incremental_run --shared-estimation \
  --shared-linearization selective --shared-preconditioner coarse \
  --shared-retry-linear-failure --contrast clahe --feature-scale 0.5 \
  --flow-recovery --map
```

`--shared-admission covisibility` selects the separate allocation experiment.
Python callers can replace `SharedConfig.relinearization` with a
`RelinearizationConfig`, including zero thresholds for exact endpoint refresh.
All policies are serialized in the online run configuration. Reproduction tools:

- `experiments/incremental_shared_study.py` freezes manifests and policies,
  launches isolated one-thread raw workers and scores paired reference roots.
- `experiments/diagnose_shared_graph.py` reports retained temporal links,
  conditional information, support lengths and reference contradictions.
  `--global-rank` performs a bounded offline dense rank audit. `--region 325 345`
  reports landmarks crossing that received-frame interval.
- `experiments/compare_shared_evidence.py` locates changes in selected features,
  native sample ownership, temporal feature pairs and causal publication. It
  also scores landmark predictions on common physical native observations.
- `experiments/benchmark_shared_solver.py --preconditioners jacobi coarse`
  compares exact-system linear steps on controlled graph families.
- `experiments/replay_shared_factors.py --preconditioner coarse` supports
  matched frozen-evidence numerical audits. `--selective` enables the cache.
  These are retrospective diagnostics, never causal tracking results.
- Existing map and horizon tools check fixed fusion cohorts, exact native
  refusion, gauge-normalized revisions and identical emitted prefixes.

## Evaluation protocol

Local ignored evidence is under `outputs/incremental_shared/`. Frozen manifests,
execution protocols, raw reports, source snapshots and driver hashes accompany
the results. The earlier `outputs/shared_estimation/verified/` evidence remains
read-only, protected by 166 file hashes. Controls in the new study link to the
complete shared backend from that study. The fresh XYZ120 reference reproduced
all saved trajectory, atlas and reconstruction arrays exactly.

The main matrix covers full TUM XYZ 792, RPY 694 and Desk 573, a static-exposure
ScanNet 160 window and full ETH3D Lab Room 263. These are five independent
recordings and 2,482 distinct inputs. Prefixes and repeated policies reuse those
inputs. Validation reuses previously studied recordings. Every worker uses
CLAHE, half-scale features, flow recovery, one numerical thread and native mapping stride eight. ETH3D uses the established 0.1 s
diagnostic reference gap. References never enter estimation. TUM and ScanNet
have no tracking losses under the evaluated policies. ETH3D retains the same
one loss and root-coverage limitation.

`incremental` means selective derivatives, coarse preconditioning and linear
failure retries with balanced observation admission. `combined` additionally
uses recent-view admission. All other settings match the reference. Separate
Desk selective-only and coarse-only runs, Desk 256 full-solver evidence admission,
RPY axial full-solver depth, XYZ120 single-method controls and RPY 240 horizon
runs isolate important interactions. The main long runs overlapped other studies,
so their timing is descriptive. They do not establish a clean throughput ratio
against the earlier archived controls.

Principal trajectory comparisons intersect each candidate with its reference
root separately. An unrelated small ablation never shrinks that population.
Full-extent maps use fixed stride-ten training views and interleaved targets,
with coverage penalties for missing predictions. Pairwise common-training maps
separate view availability from geometry. Every evaluated reconstruction is also
recomputed from saved published poses and original native samples.

## Causal accuracy and connected coverage

The following rows compare the main incremental policy with the complete
shared solver. Errors are position RMSE in millimeters on the stated common
root frames. Root availability counts received frames, including frames without
usable references. Emitted and final populations can differ.

| Recording | Final root reference → incremental | Emitted common N | Emitted error reference → incremental | Final common N | Final error reference → incremental |
|---|---:|---:|---:|---:|---:|
| XYZ 792 | 792 → 792 | 792 | 19.91 → 21.33 | 792 | 18.49 → 19.75 |
| RPY 694 | 679 → 669 | 594 | 91.67 → 106.75 | 654 | 65.71 → 79.64 |
| Desk 573 | 403 → 568 | 393 | 61.54 → 60.62 | 403 | 62.78 → 59.91 |
| ScanNet 160 | 160 → 160 | 160 | 60.42 → 63.23 | 160 | 60.52 → 63.32 |
| ETH3D 263 | 140 → 140 | 140 | 7.64 → 7.65 | 140 | 7.73 → 7.73 |

Final matched rotation RMSE changes from 0.748 to 0.798 degrees on XYZ,
3.925 to 4.864 on RPY, 3.820 to 3.657 on Desk, 1.770 to 1.825 on ScanNet and
0.299 to 0.300 on ETH3D. Desk's larger final root is a historical connection
gain. Its emitted root is 393 frames versus reference 398, so it is not a gain
in causal connected coverage or a repair of tracking losses.

Recent-view admission is not uniformly helpful. `combined` produces final
roots 792/684/398 on XYZ/RPY/Desk. Its matched final position errors are
19.08/78.76/73.36 mm, with reference errors 18.49/76.40/62.73 mm on the
corresponding populations 792/674/398. It improves RPY emitted error on its
604 common frames, 93.84 to 76.26 mm, while final position and rotation worsen.

Desk's separate policies show strong interaction with feedback:

| Desk policy | Final root | Final common N | Matched final position RMSE | Matched final rotation RMSE |
|---|---:|---:|---:|---:|
| Complete reference | 403 | 403 | 62.78 mm | 3.820 deg |
| Selective only | 568 | 403 | 66.51 mm | 4.149 deg |
| Coarse only | 573 | 403 | 82.76 mm | 5.086 deg |
| Incremental combination | 568 | 403 | 59.91 mm | 3.657 deg |

Connecting more frames does not establish better geometric estimation. A
preconditioner that solves the same linear system can alter nonlinear stopping
and subsequent tracking. These results do not support promoting any policy.

## Full reconstruction and landmark consistency

Full-map capped missing-depth error, in millimeters, changes as follows under
the main incremental policy:

| Recording | Reference → incremental | Coverage reference → incremental |
|---|---:|---:|
| XYZ 792 | 31.66 → 31.59 | 0.9653 → 0.9667 |
| RPY 694 | 87.34 → 81.69 | 0.9476 → 0.9442 |
| Desk 573 | 58.16 → 46.79 | 0.9098 → 0.9448 |
| ScanNet 160 | 54.14 → 53.53 | 0.8643 → 0.8648 |
| ETH3D 263 | 117.30 → 117.30 | 0.4422 → 0.4422 |

The full RPY score is particularly sensitive to retained training views.
The incremental root excludes training frames 330 and 340, where the complete
reference has approximately 223 and 266 mm position error in its root gauge.
On the pair's identical training population, reference error is **80.30 mm**
and incremental error is **81.69 mm**. The apparent overall improvement comes
from the changed view population and does not demonstrate better geometric
placement. The earlier complete solver's regression against the older local
control, 84.40 to 87.34 mm, therefore remains unresolved.

Desk improves on its identical training population too, 58.16 to 50.93 mm,
although coverage there changes 0.9098 to 0.9041. Its overall map benefit combines
more connected views with improved placement of the common training views.
Selective-only Desk scores 62.69 mm overall and 57.46 mm on common training.
Coarse-only Desk scores 64.86 and 69.18 mm respectively. Full-root availability
with the latter policy accompanies worse maps.

All evaluated saved reconstruction chunks match fresh fusion exactly, including
points, colors, support and scatter arrays. Exact fusion is an internal contract,
not evidence of external map accuracy. Reference-pose fusion is neither physical
surface truth nor an additive error floor.

Landmark consistency reports retain raw RGB and native residuals, reference-placed
native discrepancies, identity support and unique ownership checks. The stricter
comparison below uses identical received frame, feature index, RGB pixel and
native sample owner, with a usable point estimate and final root membership in
both policies. Values are median millimeters, complete reference → incremental.

| Recording | Common native observations | Native fit residual | Reference-placed native discrepancy |
|---|---:|---:|---:|
| XYZ | 10,019 | 4.06 → 4.07 | 13.38 → 13.55 |
| RPY | 3,129 | 3.42 → 3.37 | 22.52 → 24.44 |
| Desk | 3,785 | 2.76 → 2.46 | 36.96 → 30.99 |
| ScanNet | 1,331 | 7.29 → 7.46 | 18.84 → 18.58 |
| ETH3D | 1,485 | 3.66 → 3.66 | 8.89 → 8.89 |

RPY's improved internal fit accompanies worse external consistency even on the
same owned observations. Observations sharing a landmark are correlated, and
the identity partitions can differ between policies. These are descriptive
matched prediction errors, without an independent physical point reference or
a statistical significance claim.

## Connectivity, observability and sensor diagnosis

The reference retained graph has four consecutive keyframe pairs without shared
native-track identities on Desk and thirteen on RPY. Most optimized identity
classes have only two or three recorded views. At the final reference states,
four Desk and eight RPY conditional camera blocks are weak. The offline fresh
undamped global Schur audits find **11 null directions in dimension 666** on Desk
and **19 in dimension 792** on RPY, after fixing raw component roots and unused
cameras. These are local linearized rank results under the fixed sensor model.

RPY cameras 325, 330, 335, 340 form a concrete failure case. Their only shared
landmark with the rest of the graph is identity 4847, observed at frames 320, 325
and 340. The group can rotate collectively around that point, producing three
global null directions. All four conditional camera blocks appear well
conditioned. A graph can be connected and have healthy conditional blocks while
still lacking a determined relative pose between camera groups. A controlled
two-group synthetic test reproduces precisely this three-direction nullspace.

For a single boundary point `X_b`, the symmetry is explicit:

```text
Q_R = [R, (I - R) X_b]       Q_R X_b = X_b
      [0,       1       ]
C_i <- Q_R C_i,  X_j <- Q_R X_j  inside the camera group
```

Every internal camera-point residual and every boundary-point residual stays
unchanged for any small rotation `R`. This is missing nonlinear geometric
information, which a more accurate factorization cannot restore.

This explains why a whole-history solver can reach an excellent internal fit
and still publish inaccurate placements around frames 330 and 340. It also
explains why rank diagnostics limited to isolated camera blocks are insufficient.
The audit is offline and bounded to 2048 camera dimensions. It is not an online
global observability certificate, nor an additional likelihood or rejection rule.

Desk 256 retains up to twice as many observations per view under the complete
solver. It still ends with only 398 root frames. On 398 matched final frames,
position error improves 61.55 to 57.99 mm while rotation worsens 3.682 to 3.966
degrees. Its map error improves 58.16 to 51.76 mm, with coverage 0.9098 to 0.9028.
Increasing observation count alone does not repair absent or weak inter-group
evidence. Recent-view admission can also sacrifice useful support and regresses
the combined Desk policy.

The full RPY axial-depth control retains 679 root frames but changes their
membership. On 674 common final frames, position error worsens 76.40 to 143.18 mm.
Map error worsens 87.34 to 96.07 mm despite slightly higher rendered coverage.
Removing lateral native constraints does not solve this recording's regression.
This does not prove the native uncertainty model correct or identify a specific
calibration error.

Active-loop reference-placed native discrepancies have p95 values 14.48 mm on
Desk and 29.79 mm on RPY, with maxima 37.76 and 60.15 mm. Unjoined temporal
identity reference RMS radii have medians 4.57 and 6.20 mm. The retained evidence
does not reveal ubiquitous gross loop contradictions, but these checks cannot
exclude incorrect associations, depth binding errors, coherent calibration
bias, exposure error or reference error. No one of those causes is established.

## Numerical reference and historical revisions

On frozen full Desk evidence, another 32 complete iterations reduce Cauchy cost
5028.4942 to 4989.3971. The coarse full solver reaches the same cost to about
`1e-11`. Root-camera position RMSE stays essentially 76.28 mm under that frozen
camera-only population and gauge. Selective replay differs by about 0.0001 cost
units and gives the same external conclusion.

The saved full RPY solution is already stationary to practical precision:
another 32 complete iterations change cost 3198.7514909956 to 3198.7514905560
and root-keyframe error by only a few nanometers. Coarse and selective frozen
replays agree. Continued optimization of the same admitted objective does not
repair its full reconstruction. These camera-only scores use their first
measured camera gauge and must not be substituted for whole-trajectory scores.

With unjoined RPY identities on the same frozen raw evidence, Jacobi PCG fails
its residual check and rolls back. Coarse PCG completes 32 iterations. This is a
numerical robustness result. Its disconnected root contains only ten measured
cameras, so it cannot establish accuracy for the complete RPY trajectory.

The causal evidence comparator shows how numerical differences become different
future objectives. Desk coarse-only first differs by over 1 micrometer at
frame 37 and over 1 mm at 38. Native sample ownership first differs at 40 and
selected feature populations at 45. Its first changed consecutive-view feature
pairs are 40→45. Common RGB feature pixels remain identical. Thus fixed-ledger
parity does not imply the same subsequent tracking, conditioning or associations.
The exact frontend decision responsible for the initial divergence is unresolved.

Historical prefix deformation remains small under the new main policy. On the
same 240 root frames, RPY 240→694 revision RMSE is 2.21 mm, with p95 4.68 mm.
Prefix external error changes 70.87 to 70.60 mm. The combined policy has 3.85 mm
revision RMSE and p95 8.64 mm, with prefix error 58.40 to 56.82 mm. Both policies'
emitted prefix poses, components, source indices and versions are exactly equal
across horizons. Large errors later in the full map do not demonstrate growing
deformation of the initial 240-frame prefix.

## Solver scaling

The main incremental policy reuses approximately 82–86% of derivative rows in
Desk, ScanNet and ETH3D updates. It still reevaluates every residual and rebuilds
every normal block. The controlled chain benchmark demonstrates the separate
effect of preconditioning:

| Cameras | Jacobi PCG iterations | Coarse PCG iterations | Jacobi solve | Coarse solve |
|---|---:|---:|---:|---:|
| 32 | 59 | 23 | 0.029 s | 0.020 s |
| 128 | 136 | 24 | 0.247 s | 0.077 s |
| 512 | 189 | 30 | 1.359 s | 0.351 s |
| 2048 | 199 | 83 | 5.903 s | 2.859 s |

These single-step synthetic timings include operator and preconditioner setup
and were collected while other studies ran. All steps pass the same true
residual tolerance. At 512 chain cameras, compact blocks plus operator and coarse
preconditioner consume about 10.09 MiB, versus 71.72 MiB for a dense camera Schur
matrix alone. At 2048, counted storage is 36.75 MiB. Complete Jacobians, temporary
arrays, raw packets and Python objects are excluded from these counters.

Long shared-track graphs are a counterexample to a universal speed gain. At 512
cameras, Jacobi needs only 10 iterations and about 0.102 s, while coarse needs 5
iterations but about 0.149 s including setup. A bounded coarse space also loses
effectiveness as a chain grows, as shown by 83 iterations at 2048. The raw online
system's 512-node cap is unchanged, so the 2048 test exercises numerical scaling
beyond currently supported online graph admission.

Four alternating-order repeats of frozen real observation systems give median
linear solve times of approximately 0.133 versus 0.100 s on RPY and 0.184 versus
0.095 s on Desk. Iterations fall 172→104 and 201→76 respectively. Maximum camera
step differences are `3.3e-14` and `1.1e-6` in the left-tangent vector. Weak
directions can amplify a small residual-tolerance difference. Both methods
still satisfy the unchanged residual contract. These frozen-system benchmarks
measure numerical work and do not establish a causal accuracy benefit.

An isolated pair of complete XYZ runs, 792 frames each, on the i7-13700HX CPU gives
the following measurements with one numerical thread and reconstruction enabled:

| Measure | Complete reference | Incremental |
|---|---:|---:|
| Raw pipeline duration | 360.63 s | 277.90 s |
| Throughput | 2.20 frames/s | 2.85 frames/s |
| p95 processing latency | 2.96 s | 2.01 s |
| Peak process RSS | 316.08 MiB | 339.34 MiB |

Elapsed pipeline time falls 22.9%, with 29.8% higher throughput and 23.26 MiB
additional peak memory. This is one isolated pair, reference first, with existing
filesystem caches. It is not an order-balanced timing confidence interval.
Both policies reproduce all 79 arrays in their six corresponding prior NPZ
artifacts exactly, including raw ownership, point states, trajectory publication
and reconstruction. The reference repeats the archived complete solver, and
the incremental run repeats its main-matrix result. Export and scoring time are
excluded from pipeline timing. Windows and hosted CI were not run.

## Computational and scientific limits

This implementation is an incremental derivative cache with global residual
evaluation and temporary exact landmark elimination. It is not an incremental
Bayes-tree factorization. Residual evaluation, robust reweighting, normal-block
assembly and raw storage still grow with retained observations. The cache adds
linear memory and verifies immutable packets on every binding. Coarse assembly
also depends on landmark visibility and aggregate overlap. Its bounded dense
matrix does not make total update time constant.

The existing 512-node admission cap still applies. History is not paged to disk,
there is no recoverable out-of-core factorization, and the solver does not claim
real-time operation. Cache byte counters exclude Python dictionaries and
temporary geometry arrays. Synthetic block counters exclude complete process
memory. Raw worker peak RSS is the process-level measurement.

Covariance, calibrated sensor uncertainty, camera intrinsics, depth bias and
exposure timing are not jointly estimated. Registration confirmation can admit
an incorrect identity. Point consistency under estimated poses is an internal
fit measure. Reference-placed native samples and reference-pose fusion provide
external diagnostics, with their own sensor and reference errors. They are not
independent physical landmark or surface truth.

The evidence identifies three architectural priorities. First, retain sufficient
spatially diverse inter-group observations before temporal track support
disappears. Observation count and binary connectivity do not ensure a determined
relative pose. An adaptive camera-admission policy should respond to usable
bridge geometry, while keeping provenance and bounds on work.

Second, make exposure and binding assumptions explicit estimation state where
the data can determine them. Current exposure transforms are frozen from past
estimated motion. Historical camera revisions relinearize the camera-landmark
objective but do not revise that conditioned within-exposure motion. Joint
temporal sensor states and reversible RGB-depth bindings need independent
calibration controls and an identifiability analysis before adoption.

Third, an affected-clique factorization should handle reversible variable
partitions, component gauges and joint nullspaces directly. It must preserve
induced correlations while avoiding fabricated information in weak directions.
The complete raw solver and the matched numerical audits remain the reference
for evaluating that replacement. The current results support keeping every new
policy experimental.

The [geometric support follow-up](observable_shared_estimation.md) investigates
these collective failures with support-aware retention, reversible native
reobservation, gauge-preserving derivative transport and exposure-chart controls.
It reports complete causal results and retains the complete solver as reference.
