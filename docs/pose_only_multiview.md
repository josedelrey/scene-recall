# Experimental pose-only multiview refinement

This offline experiment refines estimated camera poses using shared RGB-D tracks.
It leaves Sparse VO, ordinary mapping, existing archives and defaults unchanged.
There are no free landmark parameters, loop closures, cross-segment constraints
or global trajectory solve. See [the measured evaluation](pose_only_evaluation.md)
for outcomes and the decision about subsequent Bundle Adjustment.

## Formulation

For camera-to-segment poses `T_i = T_SR_i`, choose one depth-supported anchor
observation `a(l)` for each track `l` in a local window. Its measured RGB optical
point `p_l` stays fixed. A target observation `j` predicts

```text
q_lj = inverse(T_j) T_a(l) p_l
e_lj = [project(K_R, q_lj) - pixel_lj, q_lj.z - depth_lj]
E(T) = 1/2 sum_l 1/(n_l - 1) sum_j rho(||W_lj e_lj||^2)
rho(s) = c^2 log(1 + s/c^2), c = 2.5
```

The depth row is omitted when the target lacks depth. The anchor must have
depth, and each window track requires at least three views and two depth samples.
The anchor's self-observation contributes no residual. Its camera pose remains
a variable unless it is the window root. Consequently these factors couple
camera poses through shared observations. Landmark world positions change only
through their anchor cameras, without independently adjusting a point or depth.

This parameterization avoids freezing world-coordinate median points generated
by the initial VO poses. Such frozen points retain initialization bias and make
camera fits largely independent. The `frozen` control deliberately fits each
camera to those initial window medians, including the anchor view so every free
camera is constrained. It is a diagnostic control, with a different factor
population and objective scale. Its absolute cost must not be ranked against
the anchored objective. The `linear` control uses the same anchor factors and
uncertainty as the primary method, replacing Cauchy with squared loss.

The robust block convention follows the standard
[robust nonlinear least-squares formulation](https://ceres-solver.readthedocs.io/latest/nnls_modeling.html#lossfunction).
The implementation uses NumPy and OpenCV, adding no solver dependency.

## Measurement uncertainty and limitations

Default image uncertainty is 1 px in each image coordinate. Depth uncertainty
is `sigma_z = 0.005 + 0.003 z^2` meters, with `z` in meters. These are explicit
engineering assumptions, without calibration against these evaluation results.

At the initial poses, propagate the anchor covariance in `(u, v, z)` through
backprojection, relative rotation and target projection. Add the target pixel
and optional depth covariance. The inverse Cholesky factor gives `W_lj`.
This retains within-factor pixel/depth correlation and accounts for anchor
depth uncertainty affecting target reprojection. Whitening stays fixed during
optimization, preserving a well-defined acceptance objective.

Residuals sharing an anchor remain correlated. Equal total weight per track
limits the influence of long tracks but does not fully model that correlation.
Overlapping windows also reuse measurements. Condition numbers are numerical
diagnostics, without a claim of posterior pose covariance. Calibration error,
RGB/depth timing skew, dynamic objects and shared depth bias remain unmodeled.
The synthetic anchor-bias probe demonstrates that the primary formulation can
still reduce residuals while making poses worse.

## Windows, gauge and solver

Training frames are selected before feature extraction. Within each independent
VO segment, process chronological windows of six training cameras with a step
of four. The windows overlap by two cameras. Fix each window's first camera
exactly and express calculations in its local coordinate frame. Every segment's
original root remains fixed. No reference pose sets or connects a gauge.

Reassess track support within each window. Choose the available depth anchor
nearest its temporal center. Select at most 256 tracks by deterministic round
robin over a 6 by 8 RGB grid, prioritizing longer tracks inside a cell. Every
camera needs at least 20 distinct selected feature identities and 15% grid
occupancy. Weak windows remain at their transported VO initialization.

Only poses are optimized, with six tangent coordinates per free camera. Analytic
Jacobians use left rigid-motion perturbations. Damped Gauss-Newton with IRLS
accepts only decreases in the exact robust objective on the same fixed factors.
There are at most 20 iterations and ten damping trials per iteration.
A refined status means that at least one decrease was accepted. It does not
certify convergence, and iteration counts remain part of the report. Reject
behind-camera proposals rather than dropping their residuals.

The first camera removes the common rigid gauge. Check the eigenvalues of the
column-normalized weighted normal matrix before each solve. Singular systems
or condition numbers above `1e8` reject the complete window. Damping is not
accepted as evidence of observability. The left correction relative to the window initialization must have translation
norm at most 10 cm and rotation at most 5 degrees. RGB-D scale is supplied by measured
depth, without using a scale alignment in fitting.

After an accepted window, carry its endpoint correction into the unseen VO tail,
preserving the tail's original relative increments. Later windows initialize
from that transported trajectory. This avoids repeatedly pulling a corrected
trajectory back toward its original global positions. Older overlap residuals
are not marginalized or jointly re-solved. Local monotonic costs therefore do
not establish a single global monotonic objective or global optimality.

Complete the full-rate trajectory by interpolating training-camera left
corrections. Translation is linear and correction rotation follows the shortest
SO(3) path, using frame-index fractions. Apply the resulting correction to the
original intermediate Sparse pose. This uses no held-out image, depth or reference.
Training endpoints bracket every segment, and no interpolation crosses a loss.
Interpolation is an additional model assumption, especially during fast motion.

## Training and evaluation isolation

The experiment defaults to segment-relative training stride 5 plus endpoints,
held-out offset 2, 24 training frames per mapping submap, 2 cm voxels and native
depth lattice stride 4. These are experimental cohort settings rather than new
mapping defaults. The split depends only on frame order and VO segment boundaries.

Tracks are rebuilt from training images and depth only. Filtering old all-view
tracks afterward would leak associations established through excluded views.
Training views alone establish feature identities, support gates, anchors,
landmarks, whitening, window selection and refinement residuals. Ground truth
is absent from the solver's `FeatureFrame` interface.

The existing full-rate Sparse initializer already consumed the original stream,
including frames later reserved as targets. Thus targets are excluded from the
new refinement and fusion stages, without claiming they were unseen by the
inherited baseline VO. Full source fingerprint verification also reads all
inputs. It checks integrity without using excluded measurements to fit poses.

After refinement is complete, reconstruct each variant from identical training
RGB-D samples using both ordinary surface voxels and projective TSDF. TSDF mesh
rendering uses the existing connected-surface renderer and 6 cm truncation.
Surface voxels use the existing 1 cm metric-disc renderer. All methods retain
the same depth filtering and evaluation lattice.

Reference-target rendering uses the target GT pose relative to the submap's
first training GT camera. It supplies an evaluation gauge without correcting
the estimated geometry. This gauge cancels common submap placement error, so
global trajectory drift must be judged separately. Unavailable training/target
references produce explicit scoring exclusions shared by every variant.

Primary scores retain every valid target sample. Report coverage, explained
fraction within 5 cm, untrimmed depth error, signed error and missing-penalized
capped errors at 5 cm, 20 cm and 1 m. Also retain baseline-fixed populations and
exact Sparse/refined common intersections separately for each representation.
Estimated-target renders are labeled as a self-consistency diagnostic. A lower
optimization cost or estimated-target error cannot establish better geometry.

## Run and reproduce

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python \
  experiments/pose_only_multiview.py \
  --odometry-run outputs/odometry/desk_sparse \
  --output-dir outputs/pose_only/desk --evaluate-maps
```

The default runs anchored Cauchy, frozen-landmark and anchored linear controls.
`--no-controls` runs only the primary method. `--window-size` and `--window-step`
change the bounded schedule. `--map-chunks 0 4` declares an explicit map-scoring
cohort. `--maps-only` evaluates already completed trajectories, refusing changed
baseline or refined inputs. Always use fresh output destinations.

Reproduce the causal counterexamples independently of datasets:

```bash
uv run --locked python experiments/pose_only_bias_probe.py \
  --output-dir outputs/pose_only/bias_probe
```

## Modules and outputs

- `multiview.pose_only` implements factor selection, whitening, analytic
  linearization, robust fitting, gauge checks and the chronological schedule.
- `multiview.pose_runner` verifies source replay, extracts training-only tracks,
  completes full-rate poses and publishes separate trajectory variants.
- `multiview.pose_evaluation` performs post-optimization map construction and
  frozen-population scoring. Reference-target rendering and target measurements remain outside fitting.

Each experiment saves the initial training `measurements.npz` tables as
`training_measurements.npz`, plus separate format-2 `anchored/`, `frozen/` and
`linear/` odometry directories. The parent report records source/trajectory
hashes, software environment, configuration, training/target split, window
support, selected track IDs, factor hashes, conditioning, accepted cost traces,
timings, exact cached-repeat checks and reference-only trajectory evaluation.

Map exports retain voxel arrays, TSDF values and meshes, training frame IDs,
target depth, every predicted depth image and per-view/pooled metrics. They
refuse changed trajectories and verify the complete canonical input stream.
All publication uses fresh staged directories. Existing outputs are untouched.
There is no automatic promotion of a refined trajectory to the default pipeline.
