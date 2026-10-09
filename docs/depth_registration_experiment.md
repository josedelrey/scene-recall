# Independent depth registration experiment

October 9, 2026. Capture: `scene0000_00`, 5578 records. Production odometry,
project dependencies, and the lock file are unchanged. No commit was created.

The independent depth objective supports the earlier discrepancy hypothesis
for 9→10, 109→110, and 189→190. Both initializations reach nearly identical
fits that agree substantially more closely with sparse VO than with the
reference. The ten-frame association also persists in three newly evaluated
ranges. Physical pose accuracy and the proposed reconstruction-boundary cause
remain unproven.

**Registration method.** The separate
`experiments/depth_registration.py` implements multiscale projective
point-to-plane ICP using NumPy and OpenCV. Point-to-plane fitting minimizes
distance along target surface normals, a standard
[ICP objective](https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html).
Organized depth and small adjacent-frame motion justify projective association
without a nearest-neighbor library. No RGB feature measurements or sparse
inlier masks enter the fit or geometric sample selection.

Depth is native z-depth in meters. Existing geometry utilities backproject it
with `K_D`, reject native 3×3 depth ranges over 5 cm, transform points, and
project into target depth. Pyramid factors are 4, 2, and 1, with original
samples at native pixel zero and intrinsics scaled consistently. Original
depth supplies point positions. A 5-pixel bilateral image supplies only
normals, with 2 cm range sigma and 2-pixel spatial sigma. Normal central
difference radii are 1, 1, and 3 pixels at the respective scales.

Correspondences require valid normals, normal agreement within 45°, and
3D distance gates of 12, 8, and 5 cm. Depth outside 0.3–6 m is excluded.
The objective uses a 1 cm Huber loss. Left pose updates use a six-variable
linearization and line search on frozen correspondences. Correspondences are
recomputed each iteration. A maximum of 35 iterations per level is explicit.
Step tolerances are 0.01 mm translation and 0.001° rotation. The solver reports
insufficient support or rank deficiency rather than inventing a constrained fit.

Native 16×16-pixel source tiles alternate between training and held-out
evaluation. Points are sampled every second pixel in each pyramid level.
Reported comparisons use source IDs supported by all four poses: VO,
reference, ICP initialized from VO, and ICP initialized from reference.
Target associations are recomputed separately for each pose. Own-support
scores and support fractions are also archived. This prevents differences
in source support from driving the reported comparisons. Target depth and
normal neighborhoods can be shared between training and evaluation, so
these held-out scores do not constitute independent sensor validation.

Forward and backward registrations are fitted independently from both pose
initializations. The forward transform maps source optical depth coordinates
to target coordinates. Its reference is `inv(T_WC_target) @ T_WC_source`.
Backward initialization uses the inverse transform. References remain
unmodified, including their small numerical deviations from rigid rotations.

**Primary quantitative results.** The ICP column uses the forward fit
initialized from VO. The reference initialization produces effectively the
same values. RMSE columns use identical held-out source support, with
pose-specific target associations, and are in millimeters.

| Transition | ICP distance to GT mm / deg | ICP distance to VO mm / deg | Held-out plane RMSE VO / GT / ICP mm | Shared support |
| --- | ---: | ---: | ---: | ---: |
| 8→9 | 4.64 / 0.157 | 6.68 / 0.209 | 5.63 / 5.37 / 5.26 | 91.9% |
| 9→10 | 40.10 / 1.547 | 6.47 / 0.192 | 6.06 / 8.63 / 5.73 | 79.3% |
| 10→11 | 0.72 / 0.059 | 2.21 / 0.095 | 6.00 / 6.03 / 5.97 | 86.7% |
| 108→109 | 4.05 / 0.254 | 3.36 / 0.123 | 7.55 / 7.78 / 7.20 | 83.3% |
| 109→110 | 25.60 / 0.754 | 6.08 / 0.171 | 8.17 / 9.20 / 7.40 | 78.8% |
| 110→111 | 7.26 / 0.181 | 10.80 / 0.293 | 9.14 / 8.43 / 7.91 | 77.7% |
| 188→189 | 8.01 / 0.288 | 4.79 / 0.161 | 6.54 / 6.65 / 6.30 | 86.2% |
| 189→190 | 29.46 / 0.850 | 6.61 / 0.232 | 6.46 / 7.96 / 6.11 | 83.6% |
| 190→191 | 6.57 / 0.142 | 5.35 / 0.193 | 6.57 / 6.18 / 5.86 | 86.6% |

![Registration comparison](/home/rlyeh/repos/scene-recall/outputs/odometry/depth_registration/registration_comparison.png)

The three suspect comparisons retain 23,315, 24,989, and 23,384 held-out
source points. ICP's own held-out support fractions are 89.1%, 88.6%, and
89.9%, versus GT's 84.0%, 84.2%, and 86.6%. Reverse common-support RMSE
VO / GT / ICP is 6.13 / 8.62 / 5.80, 8.20 / 9.25 / 7.42, and
6.51 / 8.09 / 6.14 mm. The preference persists in both directions.
The signed residual maps show a structured GT field extending across scene
surfaces, especially at 9→10, rather than a single isolated depth patch.

**Convergence and observability.** All 60 main registrations, covering
15 transitions × two initializations × two directions, reach the finest-scale
step tolerance. Thirty-six coarse or intermediate stages reach their iteration
cap, commonly with changing discrete associations. Their histories are
retained. Frozen-correspondence descent does not guarantee monotonic descent
after reprojecting and changing support.

| Suspect transition | VO-vs-GT initialization difference mm / deg | Forward/backward cycle mm / deg | Scaled Jacobian condition |
| --- | ---: | ---: | ---: |
| 9→10 | 0.000048 / 0.000001 | 0.030 / 0.0112 | 7.86 |
| 109→110 | 0.00321 / 0.000095 | 0.315 / 0.0316 | 8.05 |
| 189→190 | 0.00259 / 0.000078 | 0.305 / 0.0138 | 7.98 |

The cycle is `T_backward @ T_forward`, compared with identity. All three
weighted local Hessians have rank six. Translation columns are scaled by
median source depth, making the condition numbers comparable to rotational
columns. Their normal-Gram spectra are full rank. These are local numerical
observability checks conditional on the measured normals, not calibrated
pose covariance or proof of physical geometry quality.

Identity initialization and six seeded perturbations with per-axis standard
deviations of 20 mm translation and 1° rotation also converge to essentially
the same fit. The maximum discrepancy from the primary result is 0.020 mm
and 0.00061° across those 21 extra fits. This tests a local basin and does
not establish global uniqueness.

The held-out objective profiles along the weakest scaled Hessian direction
have minima close to the fitted pose. Plane RMSE at offset −0.02 / zero /
+0.02 is 6.54 / 5.72 / 6.72, 8.84 / 7.51 / 8.79, and
7.11 / 5.88 / 7.13 mm. The small offsets of held-out minima from zero
also show that training and held-out surfaces do not choose exactly the
same optimum. Each profile retains shared source support across its entire
sweep. Profiles assess the weakest direction locally, not every possible
nonlinear alternative.

![Weak-direction profiles](/home/rlyeh/repos/scene-recall/outputs/odometry/depth_registration/weak_mode_profiles.png)

Normal-estimation neighborhoods matter more than initialization. Increasing
the native normal radius from 3 to 5 or 9 pixels moves fits by up to
2.09 mm / 0.085°, 2.35 mm / 0.112°, and 3.99 mm / 0.159° on the three
suspect pairs. Their distances to GT remain respectively 38.05–40.10 mm,
25.59–26.35 mm, and 26.32–29.46 mm. One 5-pixel-radius variant at
189→190 hits the finest-level iteration cap. Its result is reported with
that limitation. Numerical rank remains six in these variants. The conclusion
survives this sensitivity check, while the shifts limit any accuracy claim.

**New portions and exceptions.** The experiment also registers one boundary
pair and its following control in each new range.

| Transition | ICP distance to GT mm / deg | ICP distance to VO mm / deg | Held-out plane RMSE VO / GT / ICP mm |
| --- | ---: | ---: | ---: |
| 1009→1010 | 7.27 / 0.233 | 4.20 / 0.218 | 4.36 / 4.42 / 4.11 |
| 1010→1011 | 4.42 / 0.267 | 6.32 / 0.245 | 6.10 / 4.88 / 4.61 |
| 2009→2010 | 18.84 / 0.515 | 19.31 / 0.160 | 7.17 / 7.70 / 6.29 |
| 2010→2011 | 26.43 / 0.539 | 15.91 / 0.310 | 7.68 / 7.28 / 6.18 |
| 4509→4510 | 5.00 / 0.247 | 10.42 / 0.377 | 7.63 / 5.56 / 5.14 |
| 4510→4511 | 5.63 / 0.196 | 6.80 / 0.098 | 7.35 / 6.59 / 6.37 |

At 4509→4510, GT fits depth substantially better than sparse VO.
At 2010→2011, ICP differs substantially from both poses despite good
initialization agreement and a small 0.49 mm / 0.016° forward/backward cycle.
These examples prevent a universal assignment of reference-relative error
to GT. A stable, reversible ICP fit can still disagree with physical motion.

**Ten-frame periodicity and conventions.** Three new default sparse VO runs
cover [1000, 1200), [2000, 2200), and [4500, 4700). Each tracks 199/199
transitions. The phase comparison uses absolute target source-frame indices.
Boundary means target index modulo ten equals zero.

| Source range | Boundary / other transitions | Translation RMSE boundary / other mm | Rotation RMSE boundary / other deg | Boundary rotation error exceeds mean of its two neighbors |
| --- | ---: | ---: | ---: | ---: |
| [0, 1000), existing | 99 / 900 | 13.47 / 7.59 | 0.375 / 0.177 | 79/99 |
| [1000, 1200), new | 19 / 180 | 12.89 / 8.58 | 0.354 / 0.209 | 15/19 |
| [2000, 2200), new | 19 / 180 | 16.16 / 6.68 | 0.392 / 0.141 | 18/19 |
| [3500, 3700), existing | 19 / 180 | 18.85 / 7.43 | 0.450 / 0.155 | 19/19 |
| [4500, 4700), new | 19 / 180 | 10.45 / 6.77 | 0.358 / 0.227 | 15/19 |

![Absolute-frame phase comparison](/home/rlyeh/repos/scene-recall/outputs/odometry/depth_registration/periodicity.png)

Phase zero has the largest rotation RMSE in every listed range. Mean boundary
rotation error minus the mean of its immediate neighbors is +0.126°, +0.234°,
and +0.145° in the three new ranges. This local comparison reduces the
influence of broad changes in scene difficulty. The comparisons remain
descriptive and temporally correlated, not independent statistical samples.

The indexing audit verifies contiguous zero-based local indices and
`source_frame_index = frame_index + start`. Every selected reference matrix
matches the raw `.sens` record exactly. Archived pair transforms agree with
`inv(T_SC_target) @ T_SC_source` to less than 9.3×10⁻¹⁶ in any element.
Reference-relative errors are recomputed using the same forward convention.
This provides no evidence of an indexing offset or transform inversion error.
All archived inputs match the current capture's path, size, and modification
time. Only this capture is installed locally. Cross-capture persistence
has not been tested.

**Interpretation and next experiment.** Independent depth evidence makes a
reference-pose discrepancy on the original three transitions more likely.
It substantially weakens a purely SIFT-specific explanation and a local
initialization trap for these cases. Reconstruction submap boundaries remain
a plausible cause of the periodic component. BundleFusion's public
[default submap size is ten](https://github.com/niessner/BundleFusion/blob/master/FriedLiver/zParametersBundlingDefault.txt),
while the actual reconstruction settings and intermediate poses for this
capture are unavailable. The observations do not prove that this mechanism
generated the reference errors. Depth calibration, registration artifacts,
sensor timing, surface noise, and ICP correspondence/model bias can affect
both estimators. The depth input is also shared with the reference
reconstruction, so agreement is not external ground truth.

The most informative causal next experiment is to reconstruct this capture
with controlled submap sizes, such as 8, 10, and 12, preserving local and
globally composed poses. Compare boundary relative transforms with these
depth-only measurements. Peaks that move with submap size would directly
support a reconstruction-boundary mechanism. That experiment can remain
separate from production SceneRecall estimation. A second capture would
then test whether the pattern generalizes.

**Reproduction and artifacts.** New sparse ranges can be generated into fresh
directories using the existing CLI. The current output directories already
exist and should not be overwritten.

```bash
SENS=/home/rlyeh/datasets/scannet/scans/scene0000_00/scene0000_00.sens
for start in 1000 2000 4500
do
  stop=$((start + 200))
  UV_CACHE_DIR=/tmp/scene-recall-uv-cache uv run --offline python \
    scripts/scannet_odometry.py --sens-path "$SENS" \
    --start "$start" --stop "$stop" \
    --output-dir "outputs/odometry/repeat_${start}_${stop}"
done
```

Reproduce the depth experiment, substituting fresh sparse run directories
when appropriate:

```bash
OPENBLAS_NUM_THREADS=1 UV_CACHE_DIR=/tmp/scene-recall-uv-cache \
uv run --offline python experiments/depth_registration.py \
  --sens-path /home/rlyeh/datasets/scannet/scans/scene0000_00/scene0000_00.sens \
  --runs outputs/odometry/default_0_1000 \
         outputs/odometry/default_1000_1200 \
         outputs/odometry/default_2000_2200 \
         outputs/odometry/default_3500_3700 \
         outputs/odometry/default_4500_4700 \
  --targets 9 10 11 109 110 111 189 190 191 \
            1010 1011 2010 2011 4510 4511 \
  --output-dir outputs/odometry/depth_registration_repeat
```

`outputs/odometry/depth_registration/report.json` records configuration,
code and archive hashes, source identity, phase statistics, all convergence
traces, scores, poses, sensitivity results, and objective profiles. Per-pair
NPZ files preserve both depth images, calibration, input and fitted transforms,
and held-out/common source IDs. Signed residual maps are PNG files. Outputs
remain ignored by Git. Results were repeated, and all 15 pair diagnostic
records matched exactly.

Publication-style PNG and SVG charts are generated separately by
`experiments/plot_depth_registration.py`. Matplotlib is an optional plotting
dependency, used from an existing cached environment for this run. It was
not added to project dependencies. To reproduce charts:

```bash
MPLCONFIGDIR=/tmp/scene-recall-mpl uv run --with matplotlib==3.11.2 python \
  experiments/plot_depth_registration.py outputs/odometry/depth_registration_repeat
```

The fit environment is Python 3.12.3, NumPy 2.5.3, and OpenCV 5.0.0, with
one OpenCV thread and one OpenBLAS thread. Matplotlib 3.11.2 produced the
charts. Seven numerical tests cover the finite-difference Jacobian,
native-grid calibration, training/evaluation partition, planar rank
deficiency, missing depth, known analytic motion in both directions from
two starts, and line-search descent. All 389 tests pass. The new experimental
scripts and tests pass lint and formatting checks. Charts and residual maps
were visually inspected.
