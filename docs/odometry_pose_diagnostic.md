# Sequential pose discrepancy diagnostic

Investigation of `scene0000_00` on October 9, 2026, using the current
`SparseRGBDBackend(SparseConfig())`. Production estimation was unchanged.
No commit was created. Numerical artifacts are in
`outputs/odometry/pose_diagnostic_final/` and are ignored by Git.

The leading explanation is a local discrepancy in the reconstruction reference
trajectory, plausibly associated with ten-frame submap boundaries. Ordinary
rotation–translation coupling makes a modest image discrepancy correspond to
a much larger pose discrepancy. There is also evidence of structured RGB
measurement/model mismatch, especially at 9→10. The investigation establishes
disagreement with the observations, not which pose is physically correct.

**Method and reproducibility.** The diagnostic intercepts the arrays passed to
the existing sparse pose solver and calls that solver unchanged. Each of the
15 rerun transforms matched `default_0_200/trajectory.npz` exactly, with maximum
absolute element difference zero. Source depth, matching, edge filtering,
RANSAC, refinement, and quality gates use the default settings. OpenCV uses
one thread and seed zero. These results use Python 3.12.3, NumPy 2.5.3, and
OpenCV 5.0.0.

The forward reference is `inv(T_WC_D_target) @ T_WC_D_source`. Both poses are
converted into RGB optical coordinates through the same calibration. This
capture has identity `T_RD`, so the RGB and canonical depth transforms agree.
The reference rotations are used without repair. Residuals are
`projected pixel - observed target pixel`, in native 1296×968 RGB pixels.
Positive horizontal residual means right and positive vertical means down.

Both transforms are scored on identical point IDs for every comparison:
all depth-valid matches, original RANSAC consensus, final estimator inliers,
reference 3 px threshold population, intersection, union, and the exclusive
populations. A reference threshold population is a diagnostic selection, not
an independently validated set of correct matches.

From the repository root, reproduce into a new directory:

```bash
UV_CACHE_DIR=/tmp/scene-recall-uv-cache uv run --offline python \
  experiments/diagnose_odometry.py \
  --sens-path /home/rlyeh/datasets/scannet/scans/scene0000_00/scene0000_00.sens \
  --output-dir outputs/odometry/pose_diagnostic_rerun \
  --phase-runs outputs/odometry/default_0_1000 \
               outputs/odometry/default_3500_3700
```

The archived baseline must use defaults and start at source frame zero.
The default target list covers 7→8 through 11→12, 107→108 through 111→112,
and 187→188 through 191→192. The script's image grids are specific to this
capture's RGB resolution. No additional dependencies are needed.
`diagnostics.json` records configuration, calibration, invocation, source
identity, code hashes, timestamps, populations, distributions, geometry,
and diagnostic fits. Each `pair_*.npz` preserves the exact 3D–2D inputs,
transforms, signed residuals, and masks. `rgb_controls.png` and `residuals_*.png`
provide visual inspection. Residual lines start at cyan observed pixels and
extend toward projections, magnified sixfold.

**Confirmed pose and residual discrepancies.** In the following table, both
RMSE columns use the final estimator's identical inlier population. Relative
pose errors compare the estimated transform to the reference. Coverage is
the source inlier bounding-box fraction.

| Transition | Pose error mm / deg | Matches / estimator inliers | Coverage | Estimate / reference RMSE px | Reference mean horizontal / vertical px |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7→8 | 2.01 / 0.044 | 436 / 415 | 0.913 | 0.83 / 0.99 | −0.17 / −0.31 |
| 8→9 | 3.74 / 0.075 | 231 / 194 | 0.871 | 1.32 / 1.50 | −0.06 / +0.28 |
| 9→10 | 34.37 / 1.364 | 179 / 154 | 0.837 | 1.54 / 7.11 | −0.25 / −5.72 |
| 10→11 | 2.83 / 0.083 | 185 / 168 | 0.814 | 1.24 / 1.26 | +0.06 / −0.07 |
| 11→12 | 1.05 / 0.050 | 154 / 132 | 0.769 | 1.27 / 1.34 | +0.05 / +0.19 |
| 107→108 | 11.18 / 0.275 | 134 / 123 | 0.735 | 1.06 / 1.45 | +0.32 / −0.13 |
| 108→109 | 6.36 / 0.167 | 166 / 149 | 0.714 | 1.08 / 1.37 | −0.13 / +0.47 |
| 109→110 | 26.26 / 0.657 | 141 / 127 | 0.707 | 1.15 / 2.50 | +0.58 / −0.13 |
| 110→111 | 7.90 / 0.149 | 120 / 101 | 0.702 | 1.26 / 1.44 | +0.40 / +0.12 |
| 111→112 | 2.16 / 0.021 | 110 / 93 | 0.650 | 1.03 / 1.08 | +0.19 / −0.01 |
| 187→188 | 3.52 / 0.084 | 184 / 158 | 0.859 | 1.52 / 1.69 | −0.54 / +0.07 |
| 188→189 | 3.95 / 0.135 | 252 / 229 | 0.836 | 1.19 / 1.49 | +0.70 / −0.44 |
| 189→190 | 24.48 / 0.642 | 268 / 247 | 0.889 | 1.10 / 2.93 | −1.37 / −0.57 |
| 190→191 | 11.21 / 0.324 | 259 / 218 | 0.815 | 1.48 / 1.99 | +0.31 / +0.33 |
| 191→192 | 2.27 / 0.060 | 350 / 314 | 0.914 | 1.12 / 1.20 | −0.15 / +0.11 |

The estimator's mean signed horizontal/vertical residuals on the three suspect
populations are respectively (+0.024, −0.096), (−0.002, −0.009), and
(+0.001, +0.010) px. Near-zero means do not imply unstructured residuals.
The errors already exist before LM: initial pose errors are
30.82 mm / 1.371°, 26.71 mm / 0.669°, and 23.66 mm / 0.627°.
Refinement is not the origin of these spikes.

The actual forward motions below make the compensation visible. Rotation
components are Rodrigues vectors in degrees, not Euler angles.

| Transition | Estimate translation xyz mm | Reference translation xyz mm | Estimate rotation xyz deg | Reference rotation xyz deg |
| --- | --- | --- | --- | --- |
| 8→9 | (−1.46, −3.29, +1.09) | (−2.79, −4.75, −2.09) | (+0.306, +0.086, −0.331) | (+0.254, +0.118, −0.288) |
| 9→10 | (−1.78, −5.58, +0.81) | (−0.47, +28.75, +1.68) | (+0.534, +0.173, −0.362) | (+1.805, +0.127, −0.854) |
| 10→11 | (−3.62, −8.31, −2.65) | (−2.29, −10.80, −2.39) | (+0.637, +0.203, −0.275) | (+0.567, +0.169, −0.246) |
| 109→110 | (−1.75, +8.77, −9.01) | (−11.54, −15.55, −7.47) | (−0.222, +0.266, +0.035) | (−0.788, +0.533, +0.238) |
| 189→190 | (−11.20, −0.44, −3.26) | (−29.08, +16.22, −1.81) | (+0.358, +0.521, +0.109) | (+0.848, +0.930, +0.183) |

For example, at 109→110, the reference's extra −24.3 mm vertical translation
and −0.566° x-axis rotation have largely opposing effects on vertical
projection at about 2.4 m depth. Large pose differences can therefore coexist
with modest pixel differences. Spatial and depth variation break that
compensation.

**Population selection does not account for the full discrepancy.** Each cell
below is count followed by estimate/reference RMSE in pixels. Each cell uses
one shared population for both poses.

| Transition | Original RANSAC | Estimator final | Reference threshold | Intersection | Union |
| --- | --- | --- | --- | --- | --- |
| 9→10 | 142, 1.46 / 6.92 | 154, 1.54 / 7.11 | 5, 2.48 / 2.55 | 4, 0.82 / 2.50 | 155, 1.59 / 7.09 |
| 109→110 | 126, 1.13 / 2.50 | 127, 1.15 / 2.50 | 103, 1.03 / 1.63 | 103, 1.03 / 1.63 | 127, 1.15 / 2.50 |
| 189→190 | 246, 1.10 / 2.91 | 247, 1.10 / 2.93 | 172, 1.17 / 2.01 | 168, 1.03 / 2.02 | 251, 1.19 / 2.91 |

All-match median residuals are 1.48/6.85, 0.85/1.87, and 0.83/2.48 px.
All-match RMSE is 53.29/54.14, 2.38/3.33, and 37.90/37.67 px.
Gross mismatches dominate the first and third all-match RMSE values. The
reference slightly wins the latter all-match squared objective, demonstrating
why RMSE over unfiltered descriptor matches is insufficient by itself.

Five-fold diagnostic fits split matches before fitting and evaluate every
held-out depth-valid match without residual gating. Duplicate SIFT orientations
at source locations rounded to 0.1 px stay in the same fold. Mean fold medians
for estimate/reference are 1.57/6.93, 0.90/1.84, and 0.86/2.51 px.
The estimator fit has smaller held-out residuals for about 97%, 77%, and 88%
of those matches. On 8→9 and 10→11 the corresponding mean fold medians are
1.25/1.29 and 1.03/1.00 px. These checks reduce fitting and selection bias.
They share the same SIFT and depth measurements and cannot establish physical
pose accuracy or eliminate systematic measurement bias.

**Spatial and depth evidence.** At 9→10, reference horizontal residuals change
from roughly −5.6 px in the upper-left grid cell to +4.3 px in the lower-left.
Most well-populated cells have negative reference vertical residuals, often
around −6 to −9 px. The mean horizontal value cancels much of this field.
At 109→110, reference residuals in the top quarter average (+2.19, +1.76) px,
while the bottom quarter averages (−0.68, −0.54) px. The small overall vertical
mean hides the sign-changing field. At 189→190, horizontal discrepancy grows
across the image and reverses with depth. For 1.5–2 m points the mean reference
residual is (−2.26, +0.32) px, versus (+0.86, −2.95) px at 2.5–3 m.
These fields extend across multiple scene surfaces, not an obvious isolated
bad correspondence cluster.

After fitting 9→10, mean estimator vertical residuals by image quarter are
−1.10, −1.06, +0.07, and +0.70 px. A descriptive regression using image x, y,
and inverse depth explains 34% of vertical residual variance. On 10→11 it
explains 11%. Some structure also exists on controls, including 8→9 and
190→191, so it is not exclusive to reference-boundary spikes. This confirms
measurement/model mismatch. Rolling shutter, motion blur, correlated SIFT
localization, and RGB/depth timing are hypotheses about its origin.

RGB inspection shows the same indoor surfaces through the controls, with
visible softness/blur and no obvious moving foreground object that explains
the global fields. Image appearance alone cannot establish shutter behavior
or correct timing. Both stored sensor timestamps are zero on these records.

Suspect-pair source depth median / standard deviation / 5th–95th percentile
are 1.91 / 0.29 / 1.56–2.43 m, 2.43 / 0.30 / 1.88–2.94 m, and
2.02 / 0.28 / 1.71–2.53 m. Neighboring controls have similar distributions.
The 5 cm native depth-edge filter is already active. A diagnostic uniform
source-depth scale sweep over 0.5–1.5 at the fixed reference pose gives optima
0.77, 0.98, and 1.06, with RMSE 4.57, 2.49, and 2.70 px on unchanged
estimator inliers. No single scale explains the three discrepancies, and
none reaches the fitted-pose RMSE. These are objective probes, not inferred
calibration corrections. Spatial depth bias remains possible.

Using filtered target depth for the same matched IDs also yields reverse
estimate/reference reprojection RMSE of 1.54/7.07, 1.14/2.46, and 1.08/2.92 px.
The corresponding 3D point-distance RMSE values are 12.03/17.96,
14.76/15.87, and 12.52/14.25 mm. This supplementary check does not identify
a one-direction source-depth sampling explanation. It shares SIFT selection
and should not be confused with independent depth-only registration.

**Geometry constrains the interpretation.** Centered 3D point clouds have
smallest/largest singular-value ratios of 0.325, 0.314, and 0.352, versus
0.257 and 0.282 on 8→9 and 10→11. There is no exceptional planar or collinear
degeneracy in these suspect populations. The six-column local projection
Jacobian scales translation by median source depth to make condition numbers
comparable. Conditions are 16.0, 17.2, and 15.0, versus 18.1 and 19.6 on
those controls. All six modes are observable in this local calculation.

Translation–rotation parameter correlations reach about 0.99, including on
the controls. Geometry can amplify coherent bias without an unusually bad
condition number. With an illustrative independent 1 px noise model, lateral
translation standard deviations are about 0.8–1.4 mm. A 100-replicate bootstrap
that groups duplicated SIFT source locations produces pose displacement RMSE
of 2.36 mm / 0.071°, 1.90 mm / 0.044°, and 1.13 mm / 0.029°, far below
the reference discrepancies. These conditional calculations exclude
systematic sensor error and are not calibrated uncertainty bounds.

LM initialized at either the reference or estimate, fitting exactly the same
final inlier population, converges to nearly identical diagnostic fits. The
largest translation difference between these initializations across the three
suspect pairs is under 0.001 mm. This provides no evidence for competing PnP
local minima on that population. It does not prove the fitted motion is true.

**The ten-frame association is broader than the selected examples.** Archived
default runs show elevated errors entering source frames divisible by ten.
These are descriptive comparisons of one capture, not independent samples.

| Source range | Boundary / other transitions | Boundary / other translation RMSE mm | Boundary / other rotation RMSE deg |
| --- | ---: | ---: | ---: |
| [0, 200) | 19 / 180 | 17.17 / 8.03 | 0.524 / 0.196 |
| [0, 1000) | 99 / 900 | 13.47 / 7.59 | 0.375 / 0.177 |
| [3500, 3700) | 19 / 180 | 18.85 / 7.43 | 0.450 / 0.155 |

ScanNet documents that its reconstruction uses
[BundleFusion](https://github.com/ScanNet/ScanNet#bundlefusion-reconstruction-code).
BundleFusion's public default configuration sets
[the submap size to ten](https://github.com/niessner/BundleFusion/blob/master/FriedLiver/zParametersBundlingDefault.txt).
ScanNet's processing script invokes
[a ScanNet-specific bundling configuration](https://github.com/ScanNet/ScanNet/blob/master/Server/scan_processor.py).
The actual reconstruction settings and intermediate poses for this capture
are unavailable locally. A submap-boundary reference discrepancy is therefore
a supported hypothesis, not a confirmed attribution. A temporally structured
RGB/sensor artifact could also contribute. The production backend has no
ten-frame schedule and uses only adjacent observations.

**Most likely explanation and remaining uncertainty.** The strongest evidence
supports a reconstruction-reference discrepancy at submap boundaries,
amplified by ordinary rotation–translation compensation. Supporting facts are
the repeated boundary association, coherent reference residual fields,
good neighboring agreement, similar geometry, stable diagnostic fits, and
held-out preference for the estimated projection. Random RANSAC instability,
an exceptional geometric degeneracy, a uniform depth-scale error, and an
LM-created failure have less support. A systematic RGB/depth or feature bias
remains a plausible contributor. The smaller structured residuals after fitting
9→10 support that possibility but do not explain its cause or prove the full
34 mm / 1.36° discrepancy comes from those measurements. Reference poses and
sensor observations may both contain error. No independent motion measurement
was available.

**Single most informative next experiment.** Run diagnostic bidirectional,
dense depth-only point-to-plane registration for the three suspect transitions
and their immediate controls, independently initialized from each of the two
poses. Select depth samples without SIFT or estimator inlier masks. Record
convergence, held-out geometric residuals, and observability across different
surface normals. Agreement with the sparse pose would strengthen the
reference-discrepancy explanation. Agreement with the reference would point
toward RGB feature, shutter, or synchronization bias. Initialization dependence
or weak depth geometry would leave the distinction unresolved. This experiment
removes the shared RGB correspondences that limit the current diagnostics and
can remain entirely outside the production estimator.

Validation: all 382 existing tests pass. Diagnostic residual vectors and masks
were additionally checked against OpenCV projection and threshold identities.
The diagnostic script passes the repository's formatting and lint checks.
