# Selectable RGB-D odometry

See [initial experimental results](odometry_experiments.md) for ScanNet ablations,
reproducibility checks, and measured drift. See the
[three-backend comparison](odometry_backend_comparison.md) for matched sparse,
depth-only, and hybrid runs.

Three independently selectable backends consume canonical observations and
calibration through the same sequential tracker:

| Backend | Initialization and objective | Failure behavior |
| --- | --- | --- |
| `sparse` | Existing SIFT, matching, source depth, EPnP RANSAC, LM | Failed estimation or LM starts a new segment |
| `icp` | Identity for each pair, independent multiscale depth point-to-plane ICP | Failed or unconverged geometry starts a new segment |
| `hybrid` | Complete accepted sparse pose, followed by the same ICP | Sparse failure starts a segment, ICP failure explicitly retains sparse by default |

Sparse remains the default and its estimator implementation is unchanged.
Dataset parsing stays in adapters. Reference poses are used only by evaluation.
The depth estimator uses only depth and `K_D`, regardless of RGB grids or `T_RD`.
Hybrid receives the sparse estimate after its conversion into depth frames.

## Running a capture

```bash
uv run python scripts/scannet_odometry.py \
    --sens-path ~/datasets/scannet/scans/scene0000_00/scene0000_00.sens \
    --backend hybrid --start 0 --stop 200 \
    --output-dir outputs/odometry/hybrid_example_0_200
```

The range is half-open. Omitting `--stop` processes the remaining capture.
Existing run files are never overwritten. Choose a new directory for each run.
Images are streamed, with bounded observation, feature, and pyramid caches. The CLI materializes lightweight pose records and scalar diagnostics.

The reusable API accepts any canonical RGB-D adapter:

```python
from scene_recall.odometry.evaluation import evaluate_trajectory
from scene_recall.odometry.pipeline import track_observations
from scene_recall.odometry import HybridConfig, HybridRGBDBackend

backend = HybridRGBDBackend(HybridConfig())
trajectory = list(track_observations(observations, calibration, backend))
evaluation = evaluate_trajectory(trajectory, intervals=(1, 5, 10))
```

`PairwiseBackend.estimate(source, target, calibration)` is the extension point
for all three estimators and future extensions. It returns `PairEstimate` with
a forward canonical depth-frame transform or an explicit failure. There is no backend registry or
automatic estimator selection.

## Sparse settings and quality checks

| Setting | Default | Meaning |
| --- | --- | --- |
| `ratio` | 0.75 | Forward Lowe ratio test |
| `mutual_matching` | true | Reverse nearest match must agree |
| `max_features` | 0 | Unlimited SIFT features |
| `depth_edge_threshold_m` | 0.05 | Maximum native 3x3 z-depth range |
| `iterations` | 2000 | RANSAC iteration cap |
| `confidence` | 0.999 | Target confidence, subject to the iteration cap |
| `reprojection_threshold_px` | 3.0 | Native RGB pixel threshold |
| `refine` | true | LM plus one optional fit on reclassified inliers |
| `min_correspondences` | 20 | Minimum depth-valid matches |
| `min_inliers` | 15 | Minimum final inliers |
| `min_inlier_ratio` | 0.25 | Minimum final inliers divided by correspondences |
| `min_coverage` | 0.05 | Minimum inlier bounding-box area divided by image area in both views |
| `seed` | 0 | Reset before each RANSAC call |

Target matches are unique even when mutual matching is disabled. Depth filtering
rejects borders, missing-neighbor regions, and mixed surfaces. It does not
interpolate across edges. Thresholds are configurable starting settings, rather
than calibrated sensor noise models. Bounding-box coverage does not establish
pose observability or covariance.

LM optimizes ordinary squared reprojection error on selected correspondences.
It does not apply a robust loss or jointly optimize depth. A candidate must be
finite and preserve or reduce error on its fitted population. Final inliers are
recomputed over all correspondences, with behind-camera projections rejected.
Count, ratio, and coverage gates assess the resulting consensus. Refinement
failure rejects the transition. Use `--no-refine` to evaluate unrefined estimates.

## Depth registration and hybrid safeguards

`DepthICPBackend(ICPConfig())` initializes every adjacent pair at identity.
`HybridRGBDBackend(HybridConfig())` runs the complete sparse pipeline first,
including LM and its consensus and coverage gates. It then calls
`DepthICPBackend.refine(source, target, calibration, accepted_sparse_pose)`.
The two backends share one geometric implementation. Each backend caches only
one observation's features or pyramid and is intended for serial use.

The promoted [validated experiment](depth_registration_experiment.md) supplies
native-grid subsampling at factors 4, 2, 1 with consistent depth intrinsics.
Original z-depth supplies points, while bilateral depth supplies normals.
The fixed bilateral parameters are diameter 5, range sigma 0.02 m, and spatial
sigma 2 pixels. Central normal radii are 1, 1, 3 pixels across pyramid levels.
Source points are sampled every second level pixel. Alternating native 16-pixel
tiles reserve half of those samples for diagnostics, preserving the experimental
fit population. Target normal neighborhoods can overlap those tiles.

Projective correspondences use target pixels, 45-degree normal agreement, and
3D distance gates 0.12, 0.08, 0.05 m. The Huber point-to-plane loss has scale
0.01 m. Native depth edges above 0.05 m and depths outside 0.3–6 m are excluded.
IRLS uses left pose updates and a line search over frozen correspondences.
Reassociation can change the objective population at each iteration.

`ICPConfig` exposes iteration, support, depth, Huber, normal-angle, and step
thresholds. Defaults require 30 training correspondences, at least 20% of
eligible source training samples, and rank six at every scale and after the
final update. Rank uses eigenvalues of a Jacobian whose translation columns
are scaled by median source depth. Relative eigenvalues below `1e-8`, or
absolute eigenvalues below `1e-12`, are treated as unconstrained directions.
Damping never rescues a rank-deficient pose. These checks are local numerical
observability diagnostics, not pose covariance or physical correctness proofs.

Each level allows 35 iterations. Coarse or middle iteration caps can advance
to a finer level. Acceptance requires finest-scale translation steps below
`1e-5` m and rotation steps below `0.001` degrees. Missing support, rank loss,
nonfinite updates, solver failure, stalled line search, a nonrigid candidate,
or finest-scale nonconvergence explicitly rejects the fit. Even an exact
single-plane match fails because its motion is underconstrained.

Hybrid additionally rejects corrections above 0.1 m or 5 degrees. These are
heuristic bounds on `T_ICP @ inv(T_sparse)`, with translation measured in the
target depth frame. They do not establish that smaller corrections are accurate.
`HybridConfig.refinement_failure` and `--hybrid-refinement-failure` select
`sparse` (default) or `lost`. Sparse fallback preserves the already accepted
sparse pose and reports `geometric_refinement=fallback_sparse` plus a reason.
Strict `lost` starts a new segment. A failed sparse estimate never invokes ICP.
Neither policy substitutes an unconverged candidate for an accepted estimate.

Depth diagnostics include per-level iterations and termination, final support,
rank, condition, timing, and plane RMSE. Initial and final held-out RMSE use
identical source IDs with pose-specific target associations. Missing shared
support produces null scores. Residual reduction is diagnostic and is never
interpreted as a pose-accuracy guarantee or used to choose a pose against
reference data. Hybrid acceptance does not require lower held-out residuals.

The historical depth experiment shares cloud, association, and linearization
primitives with production. It retains its original diagnostic solver, including
per-iteration traces and permissive candidate reporting. Production acceptance
is stricter than that experimental candidate protocol.

## Coordinate frames and failures

Transforms follow [camera geometry](camera_geometry.md). PnP operates in RGB
optical frames. The rig calibration converts its estimate into depth frames:

```text
T_D1D0 = inv(T_RD) @ T_R1R0 @ T_RD
T_SC_D1 = T_SC_D0 @ inv(T_D1D0)
```

For shared optical frames, RGB rays sample native depth through
`K_D @ inv(K_R)`. Separate optical frames use depth-to-RGB projection through
`T_RD` and a nearest-surface z-buffer. Discrete registration leaves holes and
approximates visibility at pixel centers. It does not rasterize surface
footprints. Native depth quality masks are applied after visibility selection,
so rejecting foreground depth cannot expose an occluded background sample.
Calibration and synchronization remain adapter responsibilities.
Sparse and hybrid require rigid `T_RD` at tolerance `1e-6` without repairing it.

The first observation initializes segment 0 at identity. Successful transitions
extend it. A failure has no accepted relative pose. Its target starts an
independent segment at identity with status `lost`. The next adjacent pair is
attempted normally. `initialized`, `tracked`, and `lost` describe the incoming
transition. Every exported `T_SC_D` needs its segment ID.

No transform relates separate segments, and reference poses never initialize a
segment. Neither export nor evaluation bridges a tracking failure. The baseline
does not implement relocalization or keyframe recovery.

## Outputs and reproducibility

- `trajectory.npz` stores frame and source indices, optional timestamps as NaN,
  segment IDs, statuses, segment-local `T_SC_D`, untouched references and their
  presence masks, accepted `T_C1C0`, backend initialization, ICP candidates, and
  hybrid RANSAC transforms. Missing transforms are all-NaN arrays. Loading
  requires no pickle.
- `frames.csv` stores failure reasons, segment-local positions, and scalar
  matching, consensus, refinement, coverage, and all backend scalar diagnostics.
- `report.json` stores configuration, calibration, input identity and range,
  software versions, thread settings, source-code fingerprint, invocation,
  tracking time,
  timestamp availability, per-frame diagnostics, and evaluation.

Input identity includes path, size, and modification time, rather than a dataset
content hash. The code fingerprint covers the Python package and run script.
The CLI fixes OpenCV to one thread and resets RANSAC's seed for each pair.
Set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` before starting Python for
controlled geometric runtime and numerical comparisons.
Poses and metrics should repeat within the same software and hardware
environment. Timing and invocation metadata differ. OpenCV versions and hardware
can change numerical results. The stateful backend is intended for serial use.

Format version 2 names the backend `sparse`, `icp`, or `hybrid`. The existing
`initial_T_C1C0` remains the RANSAC pose for sparse, is absent for independent
ICP, and is the complete accepted sparse pose for hybrid. `initial_pose_stage`
records that meaning in the report. Hybrid also saves `ransac_T_C1C0`.
`candidate_T_C1C0` stores ICP's terminal pose even on rejected refinement.
Candidates are diagnostics only and never provide trajectory continuity.
The new geometric comparison reports only successful ICP refinements, excluding
fallbacks. The generic initial/final comparison includes all accepted pairs
with an initialization and thus includes identical sparse fallback poses.

## Evaluation

References must pass finite homogeneous and rotation checks at tolerance
`1e-4`. Missing and nonrigid references are counted and excluded without repair.
They do not affect estimated trajectories.

- **RPE** compares relative transforms at exact frame intervals, defaulting to
  1, 5, and 10. Cross-segment and unavailable-reference pairs are counted
  separately. Translation is in meters and rotation in degrees.
- **Anchored error** aligns a segment at its first valid reference pose and
  measures accumulated error without further fitting. The anchor is reported.
- **Aligned ATE** fits a proper rigid position alignment within each segment.
  Metric scale stays fixed at one. At least three positions with noncollinear
  geometry are required. Otherwise alignment is unavailable.
- **Endpoint error** measures drift at the last valid reference relative to the
  segment anchor. At least two valid reference frames are required.
- **Refinement comparison** compares initial and final poses on identical
  accepted pairs. Individual errors also retain initial, RANSAC, and ICP candidate
  poses from rejected transitions when reference pairs are usable. Hybrid's
  dedicated geometric comparison counts translation and rotation improvements
  separately on successfully refined pairs.

Statistics include count, mean, median, RMSE, p95, and maximum. Empty populations
use JSON null. Tracking coverage, failures, segment count, and largest segment
length accompany accuracy. A continuous trajectory result is available only for
a single segment. Repeated resets can make segment accuracy misleading, so
always compare coverage and segment lengths too.

Runs without timestamps use frame intervals and never invent sampling times.
ScanNet poses are reconstruction references with their own errors. A benchmark
with independent measured poses remains necessary for broader accuracy claims.

## Ablations

Use the same range and quality gates while changing one setting:

```bash
uv run python scripts/scannet_odometry.py \
    --sens-path "$SENS" --stop 200 \
    --output-dir outputs/odometry/no_lm_0_200 --no-refine

uv run python scripts/scannet_odometry.py \
    --sens-path "$SENS" --stop 200 \
    --output-dir outputs/odometry/no_edges_0_200 --no-depth-edge-filter

uv run python scripts/scannet_odometry.py \
    --sens-path "$SENS" --stop 200 \
    --output-dir outputs/odometry/no_mutual_0_200 --no-mutual

uv run python scripts/scannet_odometry.py \
    --sens-path "$SENS" --stop 200 \
    --output-dir outputs/odometry/low_ransac_0_200 \
    --iterations 100 --confidence 0.99
```

The original `experiments/scannet_sift.py` preserves one-way matching, unfiltered
depth sampling, 100-iteration settings, output keys, and diagnostic-only LM.
Its raw RANSAC helper is shared with the sequential estimator.

## Controlled backend comparisons

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 UV_CACHE_DIR=/tmp/scene-recall-uv-cache \
uv run --offline python experiments/compare_odometry_backends.py \
  --sens-path "$SENS" \
  --ranges 0:200 1000:1200 2000:2200 3500:3700 4500:4700 \
  --output-dir outputs/odometry/backend_comparison_repeat
```

Multiple captures can be passed to `--sens-path`. Output must be fresh.
The runner rotates backend order across ranges and runs them serially with
one OpenCV and BLAS thread. `--summarize-only` reads existing run directories.
It verifies source identities, reference arrays, calibration, settings, code
fingerprints, and numerical environments. It reports shared continuous RPE
populations and hybrid before/after errors alongside depth residual changes.
Each individual run preserves its full trajectory, report, and scalar CSV.
`comparison.json` includes report hashes and the comparison script's hash.
Runtime is one descriptive measurement per backend and range. It includes
capture opening, indexing, decode, and tracking, excluding evaluation and export.

Optical flow, keyframes, local bundle adjustment, loop closure, motion priors,
and automatic estimator selection remain future work. Both sparse reprojection
and geometric objectives can favor biased measurements. Pairwise accuracy alone
does not resolve accumulated drift.
