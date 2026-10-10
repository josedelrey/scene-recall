# TUM RGB-D support

SceneRecall reads extracted official TUM RGB-D PNG sequences through
`scene_recall.datasets.tum.TUMReader`. Its observations, calibration, optional
reference poses, bounded materialization, and reader lifecycle follow the same
[canonical data contract](data_contract.md) as ScanNet. RGB and depth images are
decoded only when selected. Timestamp and ground-truth tables are indexed in
memory. No backend contains TUM-specific logic.

## Input and calibration

The directory must contain `rgb.txt`, `depth.txt`, and their referenced PNGs.
`groundtruth.txt` is optional. Comment lines and trailing comments are supported.
Tables must contain finite, strictly increasing timestamps with no duplicates.
Image paths must stay inside the sequence directory. Unsupported dimensions,
image types, malformed tables, and invalid quaternions raise errors. Empty
associated ranges are valid.

The [official TUM format documentation](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)
is the source for these policies:

- Preserve the native 640 × 480 RGB and registered depth PNG grids. The RGB
  array uses RGB channel order, with dtype `uint8`.
- Divide unsigned 16-bit depth by **5000**, producing `float32` z-depth in
  meters. Source zero becomes `NaN`. TUM already applied its sensor depth scale
  corrections. Applying Freiburg 1's 1.035 factor again would be incorrect.
- Use TUM's recommended ROS-default registered-image pinhole parameters:
  `fx = fy = 525`, `cx = 319.5`, `cy = 239.5`. Both canonical views have the same
  intrinsics and optical frame, with `T_RD = identity`.
- The published IR intrinsics describe the physical depth sensor, while the
  released PNG depth is registered to color. The adapter uses the registered
  color frame, with +x right, +y down, and +z forward.

The default calibration is a documented pinhole approximation. Freiburg 1 and
2 have published RGB lens distortion coefficients, and TUM explicitly
recommends the ROS-default policy because further rectification of registered
depth is difficult. The adapter preserves that policy and records it in the run
report. It does not silently apply the measured RGB intrinsics without handling
their distortion. Sensor-specific rectification would require a separately
validated adapter policy. This residual calibration uncertainty limits claims
about physical accuracy.

## Time association and references

RGB and depth timestamps are Unix epoch seconds. Association greedily accepts
candidate pairs in ascending absolute timestamp difference, without reusing
an image. Deterministic ties prefer the lower RGB index and then depth index.
The default maximum difference is 0.02 seconds, inclusive. Accepted pairs are
ordered by depth timestamp. This is a temporal association, so moving scenes
can still exhibit RGB/depth skew.

`start` and `stop` select `[start, stop)` from the complete associated-pair list.
They do not refer to either raw image table. Returned frame indices begin at
zero. Provenance retains the associated index, original RGB and depth indices,
relative filenames, original timestamps, and signed RGB-minus-depth difference.
Unmatched counts are reported for the complete source.

An observation's canonical timestamp refers to its depth view. TUM ground truth
contains `timestamp tx ty tz qx qy qz qw`, describing the **color optical camera
to motion-capture world** transform. Since registered depth shares that optical
frame, it becomes `T_WC_D` directly. No transform inversion, axis flip, or extra
camera offset is needed.

References are evaluated at the depth timestamp using linear translation and
shortest-arc quaternion SLERP. Both surrounding ground-truth samples must be
within 0.05 seconds. Exact timestamp samples are used directly. There is no
extrapolation or interpolation across an unsupported gap. Unavailable references
become `None` with a reason and never remove an RGB-D observation. Rounded unit
quaternions are normalized only within a 1e-3 norm tolerance. Other invalid
quaternions fail validation. This conversion is explicit in the adapter policy.

Ground truth never initializes or corrects odometry. It is used only for
reference evaluation. Association parameters are configurable through
`TUMConfig` and the runner's `--tum-max-rgb-depth-difference-s` and
`--tum-max-ground-truth-difference-s` flags.

## Python use

```python
from scene_recall.datasets.tum import TUMReader, load_tum_sequence
from scene_recall.odometry import SparseRGBDBackend
from scene_recall.odometry.pipeline import track_observations

with TUMReader("data/tum/rgbd_dataset_freiburg1_xyz", start=0, stop=200) as reader:
    frames = list(track_observations(reader, reader.calibration, SparseRGBDBackend()))

sequence = load_tum_sequence("data/tum/rgbd_dataset_freiburg1_xyz", stop=20)
```

## Reproducible commands

Obtain `rgbd_dataset_freiburg1_xyz.tgz` from the
[official download page](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download).
Extract it into an ignored local data directory:

```bash
mkdir -p data/tum
uv run python - <<'PY'
import tarfile

with tarfile.open("/path/to/rgbd_dataset_freiburg1_xyz.tgz") as archive:
    archive.extractall("data/tum", filter="data")
PY
```

Run a backend or a serial matched comparison with fixed default settings:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python scripts/rgbd_odometry.py \
  --tum-path data/tum/rgbd_dataset_freiburg1_xyz \
  --backend sparse --stop 200 --output-dir outputs/odometry/tum_sparse_200

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python experiments/compare_odometry_backends.py \
  --tum-path data/tum/rgbd_dataset_freiburg1_xyz \
  --ranges all --output-dir outputs/odometry/tum_xyz_repeat
```

The comparison accepts multiple TUM paths and multiple ScanNet `--sens-path`
inputs in the same invocation. Explicit `--ranges 0:200` uses the same clip
length for every input. `--ranges all` evaluates full sequences. ScanNet-only
runs retain their previous five default ranges. Any invocation containing TUM
defaults to full sequences. Use fresh output directories to preserve prior runs.
The legacy `scripts/scannet_odometry.py` delegates to the same shared runner.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python experiments/compare_odometry_backends.py \
  --sens-path /path/to/scene0000_00.sens \
  --tum-path data/tum/rgbd_dataset_freiburg1_xyz \
  --ranges 0:200 --output-dir outputs/odometry/cross_dataset_200
```

## Metrics and artifacts

Each run exports the established `trajectory.npz`, `frames.csv`, and
`report.json`, with backend settings, environment, package fingerprint,
calibration, source selection, coverage, failure reasons, stage diagnostics,
and reference poses. TUM reports additionally contain the association policy,
source counts, and per-frame provenance. A SHA-256 fingerprint covers the exact
selected canonical image arrays, reference arrays, and provenance. Matched
comparisons require identical fingerprints, source selection, calibration,
timestamps, reference poses, backend component settings, and environment.
The export schema retains version 2 with additive report fields.

Frame RPE at intervals 1, 5, and 10 remains directly comparable to the existing
ScanNet experiments. The shared evaluator also supports RPE at **0.1, 0.5, and
1.0 seconds** through `--rpe-time-intervals-s`, using a nearest later endpoint
per source within `--rpe-time-tolerance-s` (default 0.02 seconds). Ties select the
earlier target. Actual elapsed-time distributions and candidate counts are
reported. Missing or nonmonotonic clocks make time RPE unavailable while frame
RPE remains valid. The existing local ScanNet capture has unavailable timestamps.

Every tracking loss starts a separate segment. Both kinds of RPE exclude
cross-segment pairs and missing references. Comparison summaries use the same
reference-valid, continuous pair population for every backend. Anchored drift,
fixed-scale SE(3) aligned position ATE, endpoint errors, segment counts, and
largest segment size accompany coverage. Fragmented trajectories have no single
full-sequence ATE or endpoint. No similarity alignment estimates or corrects
metric scale.

These are SceneRecall's evaluation conventions. RPE uses the established
forward camera-coordinate transform `C_D0 -> C_D1`, and reference interpolation
uses the policy above. The official TUM scripts use their own time association
and relative-transform conventions. Report these settings when comparing
external scores rather than describing the results as an identical official
benchmark evaluation. Raw estimated poses remain in local segment coordinates.

See [initial cross-dataset results](tum_odometry_evaluation.md) for measurements,
limits, and next experiments. Tests run with `uv run pytest -q`,
`uv run ruff check .`, and `uv run ruff format --check .`.

The [cross-sequence evaluation](tum_cross_sequence_evaluation.md) extends the
fixed-configuration comparison to full `freiburg1_rpy` and `freiburg1_desk`,
with motion, convergence, drift, timing, and repeatability diagnostics.
