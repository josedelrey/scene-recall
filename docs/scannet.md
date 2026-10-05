# ScanNet frame extraction and point-cloud export

## Frame extraction

Load one original frame from a processed ScanNet v2 capture:

```python
from scene_recall.datasets.scannet import load_scannet_frame

sequence = load_scannet_frame(
    "~/datasets/scannet/scans/scene0000_00/scene0000_00.sens",
    original_frame_index=0,
)
observation = sequence.observations[0]
```

The adapter supports version 4 with JPEG color and zlib uint16 depth. It verifies
the expected calibrated StructureSensor registration and preserves the native
image grids. Its output follows the [RGB-D data contract](data_contract.md).

Preceding frame payloads are skipped. The original index and raw timestamps are
recorded in provenance, while the one-observation sequence has canonical frame
index 0.
Zero depth timestamps are unavailable. Finite stored reference poses with an
exact homogeneous last row `[0, 0, 0, 1]` are preserved apart from conversion to
`float64`. Rotation quality is not checked, and poses are never repaired or
projected to $SO(3)$. The all-`-inf` tracking sentinel becomes `None` with a
provenance reason. Other nonfinite poses and malformed homogeneous rows are
rejected. See `load_scannet_frame`'s docstring for supported calibration and errors.

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
