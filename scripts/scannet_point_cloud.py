"""Backproject source frame 0 to binary little-endian PLY in C_D.

Coordinates remain in the canonical depth-camera optical frame, in meters,
with +x right, +y down, and +z forward. Use --color for projected RGB samples.
No world pose is applied.
"""

import argparse
from pathlib import Path

import numpy as np

from scene_recall.datasets.scannet import load_scannet_frame
from scene_recall.geometry.camera import (
    backproject_depth,
    project_points,
    transform_points,
)


def sample_rgb_colors(
    points_D: np.ndarray, rgb: np.ndarray, K_R: np.ndarray, T_RD: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return RGB colors and a successful-sample mask without changing points_D.

    Transform C_D to C_R for projection. Accept finite projected coordinates
    within [0, W_R - 1] x [0, H_R - 1], inclusive. Nearest-neighbor sampling uses
    floor(coordinate + 0.5), so half-pixel ties round toward the larger index.
    Invalid or outside projections receive black (0, 0, 0), preserving all XYZ.
    """
    pixels = project_points(transform_points(points_D, T_RD), K_R)
    height, width = rgb.shape[:2]
    sampled = (
        np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] <= width - 1)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] <= height - 1)
    )
    indices = np.floor(pixels[sampled] + 0.5).astype(np.intp)
    colors = np.zeros((len(points_D), 3), dtype=np.uint8)
    colors[sampled] = rgb[indices[:, 1], indices[:, 0]]
    return colors, sampled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sens-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--color", action="store_true", help="include projected RGB colors"
    )
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
    records = vertices
    color_properties = ""
    if args.color:
        colors, sampled = sample_rgb_colors(
            vertices,
            sequence.observations[0].rgb,
            sequence.calibration.K_R,
            sequence.calibration.T_RD,
        )
        records = np.empty(
            len(vertices),
            dtype=[
                ("x", "<f8"),
                ("y", "<f8"),
                ("z", "<f8"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ],
        )
        for axis, name in enumerate(("x", "y", "z")):
            records[name] = vertices[:, axis]
        for channel, name in enumerate(("red", "green", "blue")):
            records[name] = colors[:, channel]
        color_properties = (
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        )
        print(f"RGB samples: {int(sampled.sum())}")
        print(f"Outside or invalid RGB projections (black): {int((~sampled).sum())}")
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment coordinate_frame C_D\n"
        "comment units meters\n"
        "comment optical_axes x_right y_down z_forward\n"
        f"element vertex {len(vertices)}\n"
        "property double x\n"
        "property double y\n"
        "property double z\n" + color_properties + "end_header\n"
    ).encode("ascii")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        stream.write(header)
        records.tofile(stream)

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
