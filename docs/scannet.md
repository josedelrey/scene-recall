# ScanNet reading and point-cloud export

## Sequential reading

Use `ScanNetReader` to process a long capture one observation at a time:

```python
from scene_recall.datasets.scannet import ScanNetReader

with ScanNetReader("/path/to/scene0000_00.sens") as reader:
    calibration = reader.calibration
    for observation in reader:
        print(observation.frame_index, observation.timestamp)
```

Construction opens one file and validates its header and shared calibration.
Iteration reads and decodes each selected record on demand, in order, without
reopening or rescanning the file. The reader retains no emitted observations.
Memory use stays bounded by the current frame unless the consumer retains data.
`reader.sequence_id` identifies the capture, and `reader.frame_count` is the
declared total source frame count.

The reader is a single-pass iterator. Calling `iter(reader)` continues its
current position. Open a new reader to replay the source. Use the context manager
for early exits, or call `close()` explicitly. The file also closes on exhaustion
and parsing, decoding, or canonical validation errors. Iterating a closed reader
produces no more observations.

## Materialized ranges and single frames

Materialize a bounded source range into the existing canonical `Sequence`:

```python
from scene_recall.datasets.scannet import load_scannet_sequence

sequence = load_scannet_sequence("/path/to/scene0000_00.sens", start=100, stop=200)
```

Both `ScanNetReader` and `load_scannet_sequence` accept the half-open range
`[start, stop)`, with `start=0` and `stop=None` by default. `None` selects through
the declared end of the capture. Bounds must be Python integers satisfying
`0 <= start <= stop <= frame_count`. Booleans are rejected. Invalid bound types
raise `TypeError`, and invalid ranges raise `IndexError`. Empty ranges are valid
and materialize as an empty `Sequence` with the source calibration.

Observation `frame_index` starts at zero within the selected range.
`provenance["original_frame_index"]` retains the position in the source, and
`provenance["source_frame_count"]` retains the total source frame count.
`Sequence` contains a materialized tuple and owns no source IO. Its memory use
grows with the selected range. Omitting `stop` can materialize the whole capture.

Load one original frame from a processed ScanNet v2 capture:

```python
from scene_recall.datasets.scannet import load_scannet_frame

sequence = load_scannet_frame(
    "~/datasets/scannet/scans/scene0000_00/scene0000_00.sens",
    original_frame_index=0,
)
observation = sequence.observations[0]
```

`load_scannet_frame` uses the same reader to materialize a one-frame range. It
retains its existing single-frame API and returns canonical frame index 0.

Preceding records have their headers and payload bounds checked once, while
their compressed payloads are skipped without reading or decoding. Records at or
after `stop` and the optional IMU stream are unread. Errors in selected frames
surface when those frames are consumed. Filesystem errors propagate as `OSError`.

## Canonical semantics

The adapter supports version 4 with JPEG color and zlib uint16 depth. It verifies
the expected calibrated StructureSensor registration and preserves the native
image grids. Its output follows the [RGB-D data contract](data_contract.md).

RGB and depth remain associated by source record. Depth is float32 z-depth in
meters, with source zero values converted to `NaN`. RGB is uint8 in RGB channel
order. Both canonical optical frames refer to the registered color optical
frame, so `T_RD` is identity despite different image grids. The world frame is
the stored reconstruction world. Companion `.txt` metadata is never read.

Raw timestamps are recorded in provenance. Nonzero depth timestamps become
seconds from source microseconds, and zero depth timestamps are unavailable.
Finite stored reference poses with an exact homogeneous last row
`[0, 0, 0, 1]` are preserved apart from conversion to `float64`.
Rotation quality is not checked, and poses are never repaired or
projected to $SO(3)$. The all-`-inf` tracking sentinel becomes `None` with a
provenance reason. Other nonfinite poses and malformed homogeneous rows are
rejected. See `ScanNetReader`'s docstring for supported calibration and errors.

## Frame 0 point-cloud export

Run from the repository root:

```bash
uv run python scripts/scannet_point_cloud.py \
    --sens-path /path/to/scene0000_00.sens \
    --output outputs/scene0000_00_frame0_C_D.ply
```

The script exports every valid depth point as float64 XYZ in binary little-endian
PLY. Coordinates remain in the canonical depth-camera frame `C_D`, following the
[camera geometry conventions](camera_geometry.md). The reference pose is unused.
No points are downsampled. The script reports point counts and the bounding box.
A frame with no valid depth pixels is rejected, and the output path must differ
from the source path.

Add `--color` to include uint8 red, green, and blue properties. For color sampling,
points are transformed from `C_D` to `C_R` using `T_RD`, then projected with `K_R`.
Exported XYZ remains in `C_D`. Finite projections within the RGB pixel-center
bounds `[0, W_R - 1]` and `[0, H_R - 1]` are sampled at the nearest pixel using
`floor(coordinate + 0.5)`, with half-pixel ties rounded toward the larger index.
Outside or invalid projections retain their XYZ and receive black. The script
reports their count. Color association assumes the projected surface is visible,
with no visibility test against an RGB depth map.

Keep source datasets outside the repository and generated point clouds under the
ignored `outputs/` directory or outside the repository. The CLI accepts both
paths explicitly.

## Two-frame SIFT experiment

Run from the repository root:

```bash
uv run python experiments/scannet_sift.py \
    --sens-path /path/to/scene0000_00.sens
```

Use `--frame0` and `--frame1` to select nonnegative source frame indices. They
default to 0 and 1. The experiment reads their enclosing range with
`ScanNetReader` and retains the selected frames in the requested order. C0 and
C1 refer to the selected source and target camera frames, respectively.
It detects SIFT on grayscale conversions of the native RGB images, then matches
source descriptors against the target using brute-force L2 KNN matching with `k=2`.
Lowe's ratio test accepts the nearest neighbor when
`nearest.distance < ratio * second_nearest.distance`. The default ratio is 0.75,
adjustable with `--ratio`. Rows with fewer than two neighbors are rejected.
Missing descriptors produce zero matches and a retention ratio of zero.

Diagnostics report keypoint counts, KNN query-row count, accepted matches, and
retention as accepted matches divided by KNN query rows. Both selected indices
must be within the capture.

Accepted source RGB pixels are backprojected with source z-depth using the
registered grids' `K_R` and `K_D`. Invalid depth removes the same row from both
pixel arrays and the 3D array. The remaining `points_C0` are float64 XYZ in
camera frame C0, in meters. Reprojecting them through `K_R` must recover their
source RGB pixels. Diagnostics also report depth-valid count and retention.

The experiment passes `points_C0`, the aligned target RGB `pixels_1`, and `K_R`
to OpenCV `solvePnPRansac`. This classical baseline uses `SOLVEPNP_EPNP`, a 3 px
reprojection threshold, 100 iterations, confidence 0.99, and `distCoeffs=None`.
It requires at least five correspondences to retain EPNP, since
[OpenCV switches to P3P with exactly four inputs](https://docs.opencv.org/4.12.0/d9/d0c/group__calib3d.html).
The returned rotation vector and translation form `T_C1C0`, mapping C0 into C1:
`points_C1 = points_C0 @ T_C1C0[:3, :3].T + T_C1C0[:3, 3]`.

Diagnostics report the correspondence count, RANSAC inlier count, inlier ratio,
estimated transform, and mean, median, RMSE, and maximum inlier reprojection
error in pixels. Errors use the final returned pose and OpenCV's returned
inlier indices. The final EPNP fit can move some returned inliers beyond the
3 px RANSAC threshold. No additional refinement or inlier filtering is applied.
Insufficient correspondences or a failed solver produce an explicit diagnostic
and preserve the correspondence and visualization outputs without an estimate.

After estimation, available ScanNet reference poses supply
`T_C1C0_gt = inv(T_WC1) @ T_WC0`. The canonical `T_WC_D` poses apply here because
ScanNet's registered depth and RGB optical frames coincide. Diagnostics print
the estimated translation, raw GT translation, both translation magnitudes,
and their vector difference norm in meters. Missing reference poses or a
singular target reference pose make the GT comparison unavailable without
discarding the estimate.

Use `--diagnose-gt-reprojection` to reproject exactly the returned RANSAC inliers
with both the estimated transform and raw relative GT, using the same `K_R`
pinhole model and observed target pixels. Each pose reports mean, median and
RMSE pixel error, mean signed residual `(du, dv)` as observed minus projected,
and the percentage with error strictly below 3 px. Nonfinite projections and
points with nonpositive target depth are counted as invalid. Their errors count
as infinity in the full selected set's statistics and as failures in the
percentage. The signed mean is unavailable if any projection is invalid.
The diagnostic only prints results and preserves the saved outputs.

Rotation comparison is local to this experiment. The raw relative GT rotation
is preserved and its Frobenius difference from the estimate is always reported
when GT is available. An angular error is reported only if `R_gt.T @ R_gt` is
within absolute tolerance `1e-4` of identity and `det(R_gt)` is within `1e-4`
of +1, both with zero relative tolerance. The angle in degrees is
`acos(clip((trace(R_est @ R_gt.T) - 1) / 2, -1, 1))`. Small departures from
rigidity within this tolerance make the angle approximate. When the checks
fail, the angle is explicitly unavailable while the translation comparison
remains available. Reference rotations are never repaired, and the canonical
data contract's transform validation policy is unchanged.

Outputs default to the ignored `outputs/` directory, adjustable with
`--output-dir`:

- `<sequence_id>_frames<frame0>_<frame1>_sift.png` shows every accepted match
  with the source frame on the left and target frame on the right. RGB colors
  are preserved.
- `<sequence_id>_frames<frame0>_<frame1>_sift.npz` contains float32 arrays
  `pixels_0` and `pixels_1`, both shaped `(N, 2)` with `(x, y)` coordinates in each frame's
  native RGB grid, plus float64 `points_C0` shaped `(N, 3)`. Row `i` in each
  array is the same depth-valid match. Empty results have shapes `(0, 2)` and
  `(0, 3)`. Matches remain directed from frame 0 to frame 1, and multiple
  frame-0 features can match the same frame-1 feature.
- The same archive includes `pose_status`, a scalar string equal to `estimated`
  on success or a failure reason otherwise. `inlier_indices` is a flat integer
  array indexing the saved correspondence rows, and
  `inlier_reprojection_errors_px` is a float64 array in that inlier order.
  These two arrays are empty when estimation is unavailable. Successful
  estimation adds float64 `T_C1C0` shaped `(4, 4)`. A successful GT comparison
  also adds raw float64 `T_C1C0_gt` shaped `(4, 4)`.

```python
import numpy as np

with np.load("outputs/scene0000_00_frames0_1_sift.npz") as matches:
    pixels_0 = matches["pixels_0"]
    pixels_1 = matches["pixels_1"]
    points_C0 = matches["points_C0"]
    if matches["pose_status"].item() == "estimated":
        T_C1C0 = matches["T_C1C0"]
        inlier_indices = matches["inlier_indices"]
```

The locked `opencv-python-headless` dependency supplies SIFT and image writing
without GUI or contrib modules. The implementation stays in `experiments/`.
