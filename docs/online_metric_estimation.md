# Online metric estimation and incremental fusion

This follow-up investigates the availability/accuracy gap exposed by the
[frontend study](online_frontend.md). Existing defaults remain unchanged. The new
metric estimator and incremental reconstruction are separately selectable.

## Diagnosis and research decisions

More accepted transitions change both the trajectory population and the path by
which error propagates. A connected pose can have small adjacent-frame error
while accumulating coherent drift. A larger fused map can improve coverage while
worsening geometry on the same training cameras. Whole-root accuracy and common
root accuracy therefore answer different questions. Disconnected gauges must
never be silently aligned together.

The implementation audit finds that historical and recovery registration fit
PnP against fixed source depths. Incoming depth only gates the result. Repeated
noisy metric anchors can give biased poses that still satisfy pixel and distance
thresholds. Half-scale detection also changes localization precision, feature
identities, landmark lifetimes and historical candidates. Better descriptors
alone do not remove those interactions. These are candidate mechanisms, rather
than proof of the cause of every previous regression.

Relevant primary research informed the scope:

- [BAD SLAM](https://openaccess.thecvf.com/content_CVPR_2019/html/Schops_BAD_SLAM_Bundle_Adjusted_Direct_RGB-D_SLAM_CVPR_2019_paper.html)
  jointly estimates dense geometry and poses, and reports sensitivity to
  synchronization and calibration. This motivates explicit sensor-model controls.
- [DROID-SLAM](https://arxiv.org/abs/2108.10869) couples learned correspondences,
  poses and depth through dense bundle adjustment. Its integration principle is
  relevant, but its published results do not validate this CPU implementation.
- [ORB-SLAM3](https://arxiv.org/abs/2007.11898) reuses co-visible observations
  and verifies place relationships across views. Repeated confirmations here
  still share observations and are not independent statistical samples.
- [MASt3R-SLAM](https://edexheim.github.io/mast3r-slam/) combines pointmap matching,
  tracking, fusion and second-order global optimization. Predicted pointmaps
  would require a distinct provenance and uncertainty contract before entering
  SceneRecall's native metric map. This work does not evaluate that model.
- [iSAM2](https://www.cs.cmu.edu/~kaess/pub/Kaess12ijrr.pdf) updates sparse
  factorizations through a Bayes tree. It is a promising replacement for repeated
  global solves, but requires preserving factor ownership and correlated local
  summaries. No iSAM2 implementation or speed comparison is claimed here.

The first controlled intervention addresses two-observation metric estimation.
A separate exact-output intervention addresses redundant reconstruction work.
Neither replaces the local landmark estimator or the global pose graph.

## Two-observation metric refinement

`--metric-refinement` adds robust generalized least squares after the existing
PnP initializer in historical loop verification, recent-anchor recovery and
optical-flow recovery. Existing feature extraction and matching stay fixed.

Only fitting rows with incoming native depth and agreement with the initial
metric gate enter refinement. Each pair contributes one vector difference
`R a + t - b`, weighted using `R C_a R.T + C_b`. Point covariance proxies propagate
one pixel of lateral uncertainty and axial scale `0.005 + 0.01 z` meters through
the camera ray Jacobian. These fixed engineering scales are not calibrated
sensor covariances. The three-dimensional residual avoids separately counting
its constituent depth and image coordinates as independent extra observations.

The solver uses a block Huber loss with threshold 2.5, at most eight IRLS
iterations, a six-variable pose normal equation and four bounded line-search
fractions. Each line search freezes its covariance. A depth-normalized rank
check rejects degenerate metric updates. This is a conditional generalized
least-squares surrogate, without a Gaussian maximum-likelihood claim or a
covariance-determinant term.

Odd withheld correspondences never choose the initialization, fitting rows,
robust weights or refinement parameters. Existing pixel, depth, spatial support
and information gates validate the final transform. Rejection remains explicit.
Without enough incoming depth, the original PnP result is still verified. No
metric depth is synthesized during a depth outage.

Measurement identities and native observations remain authoritative. No Hessian
from this conditional solve becomes an independent graph prior. Calibration,
scale bias, reused depth anchors, nearby image patches and repeated view groups
can remain correlated. More fitting pairs do not make those errors disappear.

## Transactional incremental reconstruction

`--incremental-mapping` retains one voxel accumulator for the newest chunk.
An admitted view can append when all cached prefix poses are exactly unchanged.
The implementation compares cached poses directly, in addition to consuming the
existing dirty-frame contract. It retains the original integration order and
per-frame centroid weighting.

`VoxelMap.fork()` stages a copy-on-write update. The dictionary and frame set are
copied, while cells are copied on mutation in either branch. All affected clouds
are published only after synchronization succeeds. Failure keeps the previous
clouds stale and inaccessible as a current snapshot. Retrying still checks the
cached poses, including revisions received during a failed update.

A changed pose triggers complete refusion from original camera-local samples.
Older chunks use the original rebuild path. The cache is limited to one chunk,
although trajectory, source samples and reconstructed clouds still grow with
history. Creating sorted clouds and copying voxel dictionaries remain costs.
This is incremental fusion with raw refusion after revisions, without reversible
subtraction or incremental nonlinear factorization.

## Controlled measurement evidence

`experiments/metric_noise_controls.py` runs 50 seeded matched-correspondence
controls per condition. Geometry, image noise and axial noise are fixed across
methods. These are synthetic model checks, not extra recordings.

| Condition | PnP translation median, mm | Metric translation median, mm | PnP rotation median, deg | Metric rotation median, deg |
| --- | ---: | ---: | ---: | ---: |
| Independent axial noise | 4.69 | 3.77 | 0.115 | 0.088 |
| Source depth scaled by 1.03 | 5.38 | 14.76 | 0.120 | 0.090 |
| Both depths scaled by 1.03 | 5.38 | 3.63 | 0.120 | 0.088 |

The source-only bias counterexample shows that improving one pose statistic can
worsen another under a misspecified measurement model. Native depth remains
sensor evidence, not physical ground truth. The shared-scale case still has a
metric-scale ambiguity and does not establish bias immunity.

## Reproduction

Use a fresh output directory for every run. The original online CLI also accepts
these switches independently:

```bash
uv run python -m scene_recall.slam.cli \
  --tum-path local_inputs/recording --output-dir outputs/metric_run \
  --contrast clahe --feature-scale 0.5 --flow-recovery \
  --metric-refinement --incremental-mapping --map

uv run python experiments/metric_noise_controls.py \
  --output outputs/metric_noise.json --repetitions 50
```

Raw experiments, protocols, source archives and controls live in ignored
`outputs/metric_study/`. The frozen manifests record source paths, indices,
interventions and policies. References are stripped before estimation and used
only after poses have been emitted. There is no terminal global solve.

## Complete causal evaluation

The fixed control uses CLAHE, half-scale SIFT and flow recovery. The alternative
changes only the recovery/loop registration estimator. Both use full causal
raw-stream processing, scheduled global solves and revision-aware mapping.
The separate append-fusion experiment changes only reconstruction processing.
The metric parameters were fixed before evaluating the recordings, with no
per-recording tuning afterward.

The matrix covers seven independent recordings and 4,104 distinct natural
inputs: TUM XYZ 792, RPY 694 and Pioneer 600, ScanNet 200, ETH3D Sofa 533,
Shake 285 and Desk 1,000. The ScanNet window is source indices 1000:1200.
RGB/depth outages and repeated prefixes reuse XYZ inputs and are not independent
extra data. All recordings had appeared in earlier project research. This is a
frozen-method transfer evaluation, without a claim of new unseen scenes.
Desk was held out in the preceding frontend study and is reused here.

Translation errors below use the common causal root-connected population.
Reference availability can reduce the scored subset further. They are not
retrospectively revised poses or similarities aligned across disconnected maps.

| Recording | Inputs | Common root inputs | Failed transitions, control / metric | Causal translation RMSE, mm, control / metric |
| --- | ---: | ---: | ---: | ---: |
| XYZ | 792 | 792 | 0 / 0 | 20.90 / 20.69 |
| RPY | 694 | 694 | 0 / 0 | 117.82 / 78.98 |
| Pioneer | 600 | 75 | 23 / 24 | 30.23 / 31.04 |
| ScanNet | 200 | 200 | 0 / 0 | 103.44 / 107.72 |
| Sofa | 533 | 217 | 26 / 31 | 33.63 / 70.97 |
| Shake | 285 | 7 | 135 / 140 | 15.72 / 15.72 |
| Desk | 1,000 | 770 | 2 / 5 | 50.48 / 58.88 |

Sofa's whole emitted root population changes from 262 to 230. Whole-root
translation RMSE is 44.27 versus 70.15 mm, which must not be substituted for the
common-population comparison. Shake's seven-frame common root cannot summarize
accuracy over its highly fragmented recording.

With loops disabled, both RPY local trajectories are exactly identical and
have 188.24 mm root RMSE. With loops enabled, the metric variant improves RPY
causal root RMSE by about 33%. Active constraints exceeding the reference-only
10 cm / 5 degree diagnostic fall from four to one. This ablation locates the
improvement in historical/global correction rather than more accurate local
RPY tracking. It does not prove every retained relationship is correct.

Sofa regresses even with loops disabled: its 69-frame emitted initial component
has 36.40 versus 106.48 mm RMSE. Failures increase from 26 to 31. The metric
estimator changes recent-anchor/flow recovery and the subsequent landmark
history, so its harm is not explained solely by global optimization. The
synthetic bias controls demonstrate one plausible mechanism, without proving
that a specific sensor scale error caused Sofa's measured regression.

The blank-RGB intervention at inputs 20:30 yields 11 failed transitions for both
methods and a common causal root population of 105. Root RMSE changes from
14.24 to 13.25 mm. The blank-depth intervention at 20:50 yields no tracking
failures for either method, with root RMSE 15.10 versus 15.08 mm. Existing source
measurements remain usable and incoming metric samples are never invented.

## Reconstruction evidence

Fixed reference cameras and native target depth provide a held-out geometry
check, without independent physical surface ground truth. Lower capped error is
better. Exact raw refusion is a separate internal consistency check.

| Recording | Capped surface error, mm, control / metric |
| --- | ---: |
| XYZ | 47.64 / 45.73 |
| RPY | 88.68 / 85.89 |
| Pioneer | 150.82 / 150.83 |
| ScanNet | 92.27 / 92.53 |
| Sofa | 46.92 / 49.62 |
| Shake | 151.12 / 151.12 |
| Desk, available-reference cohort | 41.73 / 44.87 |

RPY's substantial trajectory improvement gives a much smaller surface-score
improvement. The explanations include the metric being coverage-aware and the
fact that pose translation alone does not specify map quality. The experiment
does not identify an additive sensor-noise floor. Sofa uses the same 13 training
cameras for both methods, while its total connected trajectory populations
differ. Desk explicitly uses the available-reference cohort, omitting fixed
training indices 170 and 190 and retaining 22 common training views. The strict
fixed-reference protocol would withhold that map comparison. No missing-reference
camera is replaced by an estimate.

Desk also regresses with loops disabled. On 770 common causal root frames,
translation RMSE increases from 45.89 to 60.98 mm. The alternative introduces
three new failures and rescues none. Its common-root rotation changes only
slightly with loops enabled, from 4.474 to 4.456 degrees RMSE, despite worse
translation and surface scores. Active loop checks flag no 10 cm / 5 degree
violations in either method, demonstrating the limits of that threshold as a
system-level accuracy guarantee.

RPY admits more active loops with the new estimator, 170 versus 103, while
reference-scored active violations fall from four to one. Sofa also admits more
active loops, 73 versus 55, with no flagged violations in either method, yet
worsens globally. Counts and threshold acceptance do not calibrate the
information or eliminate accumulated measurement bias.

## Performance and scalability

Four separate 160-input XYZ processes ran after the concurrent accuracy and map
jobs completed, in control/append/append/control order. OpenCV and numerical
libraries used one thread. These timings include raw reading, local estimation,
place queries, scheduled global solves and mapping, and exclude model startup,
export and reference scoring.

| Mode | Total seconds | Inputs/s | Process p95, seconds | Peak RSS, MiB |
| --- | ---: | ---: | ---: | ---: |
| Original rebuild | 22.58–22.61 | 7.08–7.09 | 0.638–0.655 | 172.0–173.0 |
| Transactional append | 21.97–23.53 | 6.80–7.28 | 0.581–0.610 | 182.5–184.4 |

All four processes produce exactly identical trajectory, emitted/revised atlas,
constraint and reconstruction arrays. Append reduces p95 in this small probe,
but total throughput is mixed and memory increases. It does not establish an
end-to-end speed improvement. Sorted cloud creation, dictionary copies and pose
revisions still cost work, and global solves remain serialized.

A separate saved-final-pose fusion replay takes 3.42–3.66 s for full rebuilds
versus 2.20–2.23 s for append, with exactly equal clouds. This is an offline
component measurement, run concurrently with the accuracy matrix. It excludes
source reading and SLAM and does not measure causal correction delivery.

Both optional methods remain disabled by default. Typical-case faster processing
cannot establish a live sensor deadline or long-run bounded memory. The graph
still caps at 512 nodes, place memory at 64 entries, and historical observations,
trajectory and output clouds remain separate growing costs. An active voxel
cache is bounded by the chunk capacity, with transient staged dictionaries and
modified cells during publication.

## Verification and unresolved work

The complete Linux suite passes 1,012 tests. Lint, formatting and whitespace
checks pass. Tests cover fitting/withheld separation, depth dropout, degenerate
geometry, native-depth flow, bidirectional noiseless motion, robust fitting,
copy-on-write isolation, failed publication/retry, chunk rollover and pose
revision equivalence. Windows and hosted CI were not run.

The preceding study's 679 fingerprinted scientific artifacts are unchanged.
Fresh control trajectories, constraints and maps reproduce the earlier fast
baseline exactly. Package hashes are associated with retained source archives,
including the initial and subsequent validation-only implementation variants.
No dependency changes or automatic commits are made.

The evidence supports retaining the metric variant for controlled research.
RPY benefits, while transfer across difficult sensors and motion is mixed.
A source-dependent noise model, exposure/calibration treatment and raw shared
landmark state deserve priority over increasing acceptance thresholds. The
current per-point covariance omits shared calibration and RGB-depth registration
uncertainty. Future estimation must retain those dependencies rather than
assigning independent information to repeated source depths.

For global consistency, co-visible observation-level verification and revisable
joint geometry remain stronger architectural candidates than accumulating more
pose-only constraints. For scalability, incremental factorization, persistent
map storage and bounded asynchronous publication remain unresolved. Learned
matching and pointmap priors require their own causal, metric-provenance and
independent-scene evaluation before replacing this native RGB-D path.

The 1,000-input Desk append run reproduces all control trajectory, emitted and
revised atlas, constraint and reconstruction arrays exactly. It retains 100
admitted map views in six chunks, with 202 graph nodes and 64 bank entries. Its
standalone run takes 97.74 s with a 3.11 s maximum process latency and 308.8 MiB
peak RSS. The original full-matrix Desk timing was contended, so those totals
are not a controlled comparative speed result. This length does not reach the
512-node capacity or establish bounded lifetime storage.

Fresh 80-input metric Sofa and append XYZ prefixes reproduce the full runs'
emitted pose, component, local segment, source-index and revision-version arrays
exactly. There are 40 successful final exact-array/prefix checks, plus 45 exact
fresh-control comparisons against preceding baseline arrays. All 18 full-matrix
reconstructions pass zero-tolerance raw refusion. The final evidence manifest
retains package/source identity for all 33 raw runs.

The [conditional multiview follow-up](online_multiview_estimation.md) investigates
shared landmark variables, historical pose conditioning and revision-dependent
loop constraints across full causal pipelines.
