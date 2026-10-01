"""Backproject source frame 0 to binary little-endian XYZ PLY in C_D.

Coordinates remain in the canonical depth-camera optical frame, in meters,
with +x right, +y down, and +z forward. No world pose is applied.
"""

import argparse
from pathlib import Path

import numpy as np

from scene_recall.datasets.scannet import load_scannet_frame
from scene_recall.geometry.camera import backproject_depth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sens-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser()
    if output.resolve() == args.sens_path.expanduser().resolve():
        parser.error("--output must differ from --sens-path")

    sequence = load_scannet_frame(args.sens_path, original_frame_index=0)
    depth = sequence.observations[0].depth
    K_D = sequence.calibration.K_D
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        raise SystemExit("Frame 0 contains no valid depth pixels")

    # Choose a valid pixel near the upper-left quarter of the image, with a
    # principal-point distance of at least one quarter of the shorter dimension.
    fx, fy, cx, cy = map(float, (K_D[0, 0], K_D[1, 1], K_D[0, 2], K_D[1, 2]))
    height, width = depth.shape
    rows, columns = np.nonzero(valid)
    away = (columns - cx) ** 2 + (rows - cy) ** 2 >= (min(depth.shape) / 4) ** 2
    rows, columns = rows[away], columns[away]
    if not rows.size:
        raise SystemExit("Frame 0 contains no valid pixel sufficiently off-center")
    selected = np.argmin((columns - width // 4) ** 2 + (rows - height // 4) ** 2)
    u, v = int(columns[selected]), int(rows[selected])
    Z = float(depth[v, u])
    manual = np.array([Z * (u - cx) / fx, Z * (v - cy) / fy, Z])

    points = backproject_depth(depth, K_D)
    np.testing.assert_allclose(points[v, u], manual, rtol=1e-12, atol=1e-12)
    vertices = points[valid].astype("<f8", copy=False)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment coordinate_frame C_D\n"
        "comment units meters\n"
        "comment optical_axes x_right y_down z_forward\n"
        f"element vertex {len(vertices)}\n"
        "property double x\n"
        "property double y\n"
        "property double z\n"
        "end_header\n"
    ).encode("ascii")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        stream.write(header)
        vertices.tofile(stream)

    print("Source frame: 0")
    print("Coordinate frame: C_D, meters (+x right, +y down, +z forward)")
    print(f"Selected pixel (u, v): ({u}, {v})")
    print(f"Depth Z (m): {Z:.17g}")
    print(f"K_D (fx, fy, cx, cy): {(fx, fy, cx, cy)}")
    print("Manual formula: (Z * (u - cx) / fx, Z * (v - cy) / fy, Z)")
    print(f"Manual XYZ (m): {manual.tolist()}")
    print(f"backproject_depth XYZ (m): {points[v, u].tolist()}")
    print(f"Exported points: {len(vertices)}")
    print(f"Removed invalid depth pixels: {int((~valid).sum())}")
    print(f"Bounding box minimum XYZ (m): {vertices.min(axis=0).tolist()}")
    print(f"Bounding box maximum XYZ (m): {vertices.max(axis=0).tolist()}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
