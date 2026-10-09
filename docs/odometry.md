# Sparse RGB-D visual odometry

See [initial experimental results](odometry_experiments.md) for ScanNet ablations,
reproducibility checks, and measured drift.

The sequential baseline detects SIFT features, matches descriptors, samples
source depth, estimates forward motion with EPnP RANSAC, and optionally refines
it with LM. It consumes canonical observations and calibration. Dataset parsing
stays in adapters. Reference poses are used only by evaluation.

## Running a capture

```bash
uv run python scripts/scannet_odometry.py \
    --sens-path ~/datasets/scannet/scans/scene0000_00/scene0000_00.sens \
    --start 0 --stop 200 \
    --output-dir outputs/odometry/default_0_200
```

The range is half-open. Omitting `--stop` processes the remaining capture.
Existing run files are never overwritten. Choose a new directory for each run.
Images are streamed, and only adjacent observations and their feature cache are
retained. The CLI materializes lightweight pose records and scalar diagnostics.

The reusable API accepts any canonical RGB-D adapter:

```python
from scene_recall.odometry.evaluation import evaluate_trajectory
from scene_recall.odometry.pipeline import track_observations
from scene_recall.odometry.sparse import SparseConfig, SparseRGBDBackend

backend = SparseRGBDBackend(SparseConfig())
trajectory = list(track_observations(observations, calibration, backend))
evaluation = evaluate_trajectory(trajectory, intervals=(1, 5, 10))
```

`PairwiseBackend.estimate(source, target, calibration)` is the extension point
for future estimators. It returns `PairEstimate` with a forward canonical
depth-frame transform or an explicit failure. There is no backend registry or
automatic estimator selection.

## Settings and quality checks

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
The backend requires rigid `T_RD` at tolerance `1e-6` without repairing it.

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
  presence masks, accepted `T_C1C0`, and initial RANSAC transforms. Missing
  transforms are all-NaN arrays. Loading requires no pickle.
- `frames.csv` stores failure reasons, segment-local positions, and scalar
  matching, consensus, refinement, and coverage diagnostics.
- `report.json` stores configuration, calibration, input identity and range,
  software versions, source-code fingerprint, invocation, tracking time,
  timestamp availability, per-frame diagnostics, and evaluation.

Input identity includes path, size, and modification time, rather than a dataset
content hash. The code fingerprint covers the Python package and run script.
The CLI fixes OpenCV to one thread and resets RANSAC's seed for each pair.
Poses and metrics should repeat within the same software and hardware
environment. Timing and invocation metadata differ. OpenCV versions and hardware
can change numerical results. The stateful backend is intended for serial use.

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
  accepted pairs. Individual errors also retain initial RANSAC poses from
  rejected transitions when reference pairs are usable.

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

Optical flow, 3D–3D registration, robust-loss refinement, keyframes, local bundle
adjustment, loop closure, and motion priors remain future work. The current
fitting objective uses source depth and image correspondences. Low reprojection
error can still conceal biased or poorly constrained motion.
