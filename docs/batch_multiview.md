# Offline batch multiview refinement

Batch RGB-D refinement fits camera poses and shared landmarks across an existing
training-track graph. It is an opt-in offline experiment. Sparse remains the
default odometry backend.

## Prepare and run

Start with a Sparse odometry run whose source provides known RGB and depth
acquisition timestamps. Prepare training-only measurements using the
[temporal fixed-landmark workflow](temporal_fixed_landmarks.md). The batch runner
consumes its verified numeric packets without reopening RGB-D media.

Use fresh output directories:

```bash
uv run --locked python experiments/temporal_fixed_landmarks.py \
  --odometry-run outputs/odometry/sequence \
  --output-dir outputs/temporal/sequence

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python \
  experiments/batch_multiview.py \
  --fixed-run outputs/temporal/sequence \
  --output-dir outputs/batch/sequence --evaluate-maps
```

Use `--map-chunks` to select map chunks, or `--maps-only` to evaluate an existing
batch run. `--map-output-dir` writes a separate map evaluation. The batch output
includes unchanged clock-only and temporal fixed-landmark controls.

Replay fitting independently from saved packets and summarize external scores:

```bash
uv run --locked python experiments/replay_batch_packets.py \
  outputs/batch/sequence --output outputs/batch/replay.json

uv run --locked python experiments/summarize_batch.py \
  outputs/batch/sequence --output outputs/batch/summary.json
```

`--depth-model axial` enables a matched depth-model ablation in a separate run.
`--min-track-views` and `--max-iterations` override the fitting defaults.

## Method and safeguards

One 3D point persists per admitted training identity throughout its original
segment. The original segment root stays fixed. The native model uses RGB 2D
and native-depth 3D residuals, frozen exposure transforms and covariance,
separate block Cauchy losses, and inverse-track-length weights. Points initialize
from geometric medians of measured depth transformed by Sparse. Reference poses
and saved world points never initialize landmarks or enter fitting.

The axial ablation retains the same selected observations and initial points,
replacing the native 3D residual with a z residual normalized by its marginal
native z standard deviation.

Default support and solver limits are:

| Setting | Default |
| --- | ---: |
| Remaining views per track | 3 |
| Native depth samples per track | 2 |
| Observations per admitted camera | 12 |
| Occupied cells on the 6 by 8 image grid | 15% |
| Maximum cameras per segment | 200 |
| Maximum iterations | 40 |
| Maximum normalized undamped condition | `1e8` |
| Maximum camera-center and point movement | 0.5 m |
| Maximum orientation change | 30 degrees |
| Cauchy loss scale | 2.5 |

Camera and track pruning repeats until support is stable. Compact analytic
Jacobians and independent 3 by 3 point elimination produce a dense camera Schur
system. Undamped rank checks reject disconnected or ill-conditioned geometry.
Damping and additional component roots do not supply missing observability.
A rejected segment receives no batch update and equals the clock-only control.

Accepted corrections interpolate across unsupported training cameras and carry
beyond the last supported camera. Full-rate completion converts poses back to
depth exposure times while preserving source metadata, losses and independent
segment roots. Unknown acquisition clocks are rejected.

## Reconstruction evaluation

Map evaluation fuses original native training depth using surface voxels and
TSDF meshes. Optimized landmarks do not replace depth samples. Reference target
poses supply external reconstruction scores, with coverage, missing-prediction
penalties and exact common-pixel comparisons retained separately. See
[held-out validation](heldout_mapping_validation.md) for scoring conventions.

Evaluate a different reconstruction extent without changing fitted trajectories
or training observations:

```bash
uv run --locked python experiments/evaluate_batch_horizon.py \
  --batch-run outputs/batch/sequence \
  --output-dir outputs/batch/long_maps --map-budget 160
```

An optional `--joint-run` adds a compatible
[local joint BA control](local_rgbd_ba.md). Different map budgets change target
chunk membership and must be reported as separate evaluation populations.

By default, map evaluation requires references for all training cameras.
`--allow-partial-training-reference` permits reference-target rendering when the
first training anchor and each scored target have valid references. All native
training samples remain in fusion, missing references are never imputed, and
incomplete reference training clouds remain labelled unpaired.

## Outputs and limitations

Each run saves trajectory reports, numeric training measurements, native depth
bindings, factor fingerprints, initial and fitted poses, shared points and solver
diagnostics. Input hashes are checked before publication. Use separate storage
for ignored outputs and datasets when transferring work between machines.

This method requires supported persistent geometry and uses future observations.
It is bounded offline refinement and does not implement online tracking or loop
closure. Timing, calibration, correlated measurements and coherent depth bias
remain limitations. Known timestamps are acquisition proxies rather than
calibrated exposure midpoints.

Target images and depth are excluded from new refinement and fusion. The original
full-rate Sparse initializer already used the source stream. Lower objectives or
better aligned position error alone do not establish better reconstruction.
Compare orientation, drift, coverage and external map geometry as well.
