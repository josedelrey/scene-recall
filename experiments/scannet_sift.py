"""Match SIFT features between source RGB frames 0 and 1 of a ScanNet capture."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from scene_recall.datasets.scannet import ScanNetReader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sens-path", type=Path, required=True)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    if not 0 < args.ratio < 1:
        parser.error("--ratio must be finite and strictly between 0 and 1")

    with ScanNetReader(args.sens_path, start=0, stop=2) as reader:
        rgb_0 = next(reader).rgb
        rgb_1 = next(reader).rgb
        sequence_id = reader.sequence_id

    sift = cv2.SIFT_create()
    keypoints_0, descriptors_0 = sift.detectAndCompute(
        cv2.cvtColor(rgb_0, cv2.COLOR_RGB2GRAY), None
    )
    keypoints_1, descriptors_1 = sift.detectAndCompute(
        cv2.cvtColor(rgb_1, cv2.COLOR_RGB2GRAY), None
    )
    knn_matches = []
    if descriptors_0 is not None and descriptors_1 is not None:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        knn_matches = matcher.knnMatch(descriptors_0, descriptors_1, k=2)
    matches = [
        neighbors[0]
        for neighbors in knn_matches
        if len(neighbors) == 2
        and neighbors[0].distance < args.ratio * neighbors[1].distance
    ]

    # Both arrays follow the same accepted-match order, including empty results.
    pixels_0 = np.array(
        [keypoints_0[match.queryIdx].pt for match in matches], dtype=np.float32
    ).reshape(-1, 2)
    pixels_1 = np.array(
        [keypoints_1[match.trainIdx].pt for match in matches], dtype=np.float32
    ).reshape(-1, 2)

    retention = len(matches) / len(knn_matches) if knn_matches else 0.0
    print("Source RGB frames: 0 -> 1")
    print(f"Keypoints in frame 0: {len(keypoints_0)}")
    print(f"Keypoints in frame 1: {len(keypoints_1)}")
    print(f"KNN matches (query rows, k=2): {len(knn_matches)}")
    print(f"Matches surviving ratio test ({args.ratio:g}): {len(matches)}")
    print(f"Retention ratio (accepted / KNN rows): {retention:.4f}")

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{sequence_id}_frames0_1_sift"
    coordinates_path = output_dir / f"{stem}.npz"
    np.savez(coordinates_path, pixels_0=pixels_0, pixels_1=pixels_1)

    # OpenCV draws and writes BGR images. Keep the native RGB pixel grids.
    visualization = cv2.drawMatches(
        cv2.cvtColor(rgb_0, cv2.COLOR_RGB2BGR),
        keypoints_0,
        cv2.cvtColor(rgb_1, cv2.COLOR_RGB2BGR),
        keypoints_1,
        matches,
        None,
        matchColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    visualization_path = output_dir / f"{stem}.png"
    if not cv2.imwrite(str(visualization_path), visualization):
        raise SystemExit(f"Could not write visualization: {visualization_path}")
    print(f"Matched RGB pixels (x, y): {pixels_0.shape}, {pixels_1.shape}")
    print(f"Coordinates: {coordinates_path}")
    print(f"Visualization (frame 0 left, frame 1 right): {visualization_path}")


if __name__ == "__main__":
    main()
