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

The experiment reads source frames 0 and 1 with `ScanNetReader(start=0, stop=2)`.
It detects SIFT on grayscale conversions of the native RGB images, then matches
frame 0 descriptors against frame 1 using brute-force L2 KNN matching with `k=2`.
Lowe's ratio test accepts the nearest neighbor when
`nearest.distance < ratio * second_nearest.distance`. The default ratio is 0.75,
adjustable with `--ratio`. Rows with fewer than two neighbors are rejected.
Missing descriptors produce zero matches and a retention ratio of zero.

Diagnostics report keypoint counts, KNN query-row count, accepted matches, and
retention as accepted matches divided by KNN query rows. A capture must contain
at least two frames. The experiment uses only RGB from the reader's observations.
Depth association and pose estimation are outside its scope.

Outputs default to the ignored `outputs/` directory, adjustable with
`--output-dir`:

- `<sequence_id>_frames0_1_sift.png` shows every accepted match with frame 0 on
  the left and frame 1 on the right. RGB colors are preserved.
- `<sequence_id>_frames0_1_sift.npz` contains float32 arrays `pixels_0` and
  `pixels_1`, both shaped `(N, 2)` with `(x, y)` coordinates in each frame's
  native RGB grid. Row `i` in each array is the same accepted match. Empty
  results have shape `(0, 2)`. Matches remain directed from frame 0 to frame 1,
  and multiple frame 0 features can match the same frame 1 feature.

```python
import numpy as np

with np.load("outputs/scene0000_00_frames0_1_sift.npz") as matches:
    pixels_0 = matches["pixels_0"]
    pixels_1 = matches["pixels_1"]
```

The locked `opencv-python-headless` dependency supplies SIFT and image writing
without GUI or contrib modules. The implementation stays in `experiments/`.
