"""Estimate a two-frame ScanNet pose from depth-valid SIFT matches."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from scene_recall.datasets.scannet import ScanNetReader
from scene_recall.geometry.camera import backproject_rgb_pixels, project_points


def estimate_relative_pose(
    points_C0: np.ndarray, pixels_1: np.ndarray, K_R: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return T_C1C0, correspondence inlier indices, and inlier errors in pixels."""
    # OpenCV switches to P3P with exactly four inputs. Keep this baseline EPNP.
    if len(points_C0) < 5:
        raise ValueError("EPNP RANSAC requires at least 5 correspondences")
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=np.ascontiguousarray(points_C0, dtype=np.float64),
        imagePoints=np.ascontiguousarray(pixels_1, dtype=np.float64),
        cameraMatrix=K_R,
        distCoeffs=None,
        iterationsCount=100,
        reprojectionError=3.0,
        confidence=0.99,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or inliers is None or inliers.size == 0:
        raise RuntimeError("solvePnPRansac did not return a pose with inliers")
    if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
        raise RuntimeError("solvePnPRansac returned a nonfinite pose")

    # Object frame is C0 and the target image is C1, so no inversion is needed.
    T_C1C0 = np.eye(4, dtype=np.float64)
    T_C1C0[:3, :3] = cv2.Rodrigues(rvec)[0]
    T_C1C0[:3, 3] = tvec.reshape(3)
    inlier_indices = inliers.reshape(-1)
    projected, _ = cv2.projectPoints(points_C0[inlier_indices], rvec, tvec, K_R, None)
    errors = np.linalg.norm(projected.reshape(-1, 2) - pixels_1[inlier_indices], axis=1)
    if not np.isfinite(errors).all():
        raise RuntimeError("PnP inlier reprojection errors are nonfinite")
    return T_C1C0, inlier_indices, errors


def print_reprojection_diagnostics(
    label: str,
    points_C0: np.ndarray,
    pixels_1: np.ndarray,
    K_R: np.ndarray,
    T_C1C0: np.ndarray,
) -> None:
    """Report a pose on the unchanged RANSAC-selected correspondence set."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        points_C1 = points_C0 @ T_C1C0[:3, :3].T + T_C1C0[:3, 3]
        projected = project_points(points_C1, K_R)
        residuals = pixels_1 - projected
        valid = np.isfinite(residuals).all(axis=1)
        # Invalid projections remain failures in the full selected population.
        errors = np.hypot(residuals[:, 0], residuals[:, 1])
        errors[~valid] = np.inf
        mean = errors.mean()
        median = np.median(errors)
        rmse = np.sqrt(np.mean(errors**2))
        signed_mean = residuals.mean(axis=0) if valid.all() else None
    print(f"{label} inlier reprojection: selected={len(errors)}, invalid={sum(~valid)}")
    if not valid.all():
        print(f"{label} invalid reprojection errors count as +inf in statistics")
    print(
        f"{label} inlier reprojection errors (px): "
        f"mean={mean:.4f}, median={median:.4f}, RMSE={rmse:.4f}"
    )
    signed_summary = (
        f"du={signed_mean[0]:.4f}, dv={signed_mean[1]:.4f}"
        if signed_mean is not None
        else "unavailable (invalid projections)"
    )
    print(f"{label} mean signed residual (observed - projected, px): {signed_summary}")
    print(
        f"{label} reprojection error below 3 px (all selected): "
        f"{100 * np.count_nonzero(errors < 3) / len(errors):.2f}%"
    )


def diagnose_pnp_refinement(
    points_C0: np.ndarray,
    pixels_1: np.ndarray,
    K_R: np.ndarray,
    T_C1C0: np.ndarray,
    T_C1C0_gt: np.ndarray | None,
) -> None:
    """Compare LM and the original pose on the original RANSAC inliers only."""
    poses = [("Original EPnP+RANSAC", T_C1C0)]
    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            objectPoints=np.array(points_C0, dtype=np.float64, order="C", copy=True),
            imagePoints=np.array(pixels_1, dtype=np.float64, order="C", copy=True),
            cameraMatrix=K_R.copy(),
            distCoeffs=None,
            rvec=cv2.Rodrigues(T_C1C0[:3, :3].copy())[0],
            tvec=T_C1C0[:3, 3].reshape(3, 1).copy(),
        )
        if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
            raise RuntimeError("solvePnPRefineLM returned a nonfinite pose")
        refined = np.eye(4, dtype=np.float64)
        refined[:3, :3] = cv2.Rodrigues(rvec)[0]
        refined[:3, 3] = tvec.reshape(3)
    except (ValueError, RuntimeError, cv2.error) as error:
        print(f"PnP refinement failed: {error}")
    else:
        print("PnP refinement: solvePnPRefineLM on original RANSAC inliers")
        poses.append(("LM-refined", refined))

    valid_gt_rotation = False
    if T_C1C0_gt is not None:
        poses.append(("GT", T_C1C0_gt))
        R_gt = T_C1C0_gt[:3, :3]
        valid_gt_rotation = np.allclose(
            R_gt.T @ R_gt, np.eye(3), rtol=0, atol=1e-4
        ) and np.isclose(np.linalg.det(R_gt), 1, rtol=0, atol=1e-4)
    else:
        print("PnP refinement GT comparison unavailable: no relative GT pose")
    for label, pose in poses:
        print_reprojection_diagnostics(label, points_C0, pixels_1, K_R, pose)
        if T_C1C0_gt is None:
            print(f"{label} translation-vector error against GT (mm): unavailable")
            print(f"{label} rotation error against GT (deg): unavailable")
            continue
        translation_error_mm = 1000 * np.linalg.norm(pose[:3, 3] - T_C1C0_gt[:3, 3])
        print(
            f"{label} translation-vector error against GT (mm): "
            f"{translation_error_mm:.6f}"
        )
        if valid_gt_rotation:
            cosine = (np.trace(pose[:3, :3] @ R_gt.T) - 1) / 2
            angle = (
                0.0 if label == "GT" else np.degrees(np.arccos(np.clip(cosine, -1, 1)))
            )
            print(f"{label} rotation error against GT (deg): {angle:.6f}")
        else:
            print(
                f"{label} rotation error against GT (deg): unavailable "
                "(raw GT fails rotation check at atol=1e-4)"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sens-path", type=Path, required=True)
    parser.add_argument("--frame0", type=int, default=0)
    parser.add_argument("--frame1", type=int, default=1)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--diagnose-gt-reprojection", action="store_true")
    parser.add_argument("--diagnose-pnp-refinement", action="store_true")
    args = parser.parse_args()
    if not 0 < args.ratio < 1:
        parser.error("--ratio must be finite and strictly between 0 and 1")
    if args.frame0 < 0 or args.frame1 < 0:
        parser.error("--frame0 and --frame1 must be nonnegative source indices")

    start = min(args.frame0, args.frame1)
    stop = max(args.frame0, args.frame1) + 1
    with ScanNetReader(args.sens_path, start=start, stop=stop) as reader:
        frames = {
            index: frame
            for index, frame in enumerate(reader, start=start)
            if index in (args.frame0, args.frame1)
        }
        calibration = reader.calibration
        sequence_id = reader.sequence_id
    frame_0, frame_1 = frames[args.frame0], frames[args.frame1]
    rgb_0, rgb_1 = frame_0.rgb, frame_1.rgb

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

    # ScanNet's registered RGB and depth grids share optical frame C0.
    points_C0 = backproject_rgb_pixels(
        pixels_0, frame_0.depth, calibration.K_R, calibration.K_D
    )
    valid_depth = np.isfinite(points_C0).all(axis=1)
    points_C0 = points_C0[valid_depth]
    pixels_0 = pixels_0[valid_depth]
    pixels_1 = pixels_1[valid_depth]
    if not np.allclose(
        project_points(points_C0, calibration.K_R),
        pixels_0,
        rtol=1e-10,
        atol=1e-8,
    ):
        raise RuntimeError(f"Frame-{args.frame0} RGB reprojection check failed")

    retention = len(matches) / len(knn_matches) if knn_matches else 0.0
    depth_retention = len(points_C0) / len(matches) if matches else 0.0
    print(f"Source RGB frames: {args.frame0} -> {args.frame1}")
    print(f"Keypoints in frame {args.frame0}: {len(keypoints_0)}")
    print(f"Keypoints in frame {args.frame1}: {len(keypoints_1)}")
    print(f"KNN matches (query rows, k=2): {len(knn_matches)}")
    print(f"Matches surviving ratio test ({args.ratio:g}): {len(matches)}")
    print(f"Retention ratio (accepted / KNN rows): {retention:.4f}")
    print(f"Matches with valid frame-{args.frame0} depth: {len(points_C0)}")
    print(f"Depth-valid retention ratio (valid / accepted): {depth_retention:.4f}")
    print(f"Frame-{args.frame0} RGB reprojection check: passed")

    results = {"points_C0": points_C0, "pixels_0": pixels_0, "pixels_1": pixels_1}
    print(f"3D-2D correspondences: {len(points_C0)}")
    print("PnP RANSAC: EPNP, 3 px threshold, 100 iterations, confidence 0.99")
    try:
        T_C1C0, inlier_indices, errors = estimate_relative_pose(
            points_C0, pixels_1, calibration.K_R
        )
    except (ValueError, RuntimeError, cv2.error) as error:
        print(f"Pose estimation unavailable: {error}")
        print("RANSAC inliers: 0")
        print("Inlier ratio (inliers / correspondences): 0.0000")
        print("Inlier reprojection errors: unavailable")
        print("GT comparison unavailable: no estimated pose")
        if args.diagnose_gt_reprojection:
            print("GT reprojection diagnostic unavailable: no estimated pose")
        if args.diagnose_pnp_refinement:
            print("PnP refinement unavailable: no estimated pose")
        results["pose_status"] = np.array(str(error))
        results["inlier_indices"] = np.empty(0, dtype=np.int32)
        results["inlier_reprojection_errors_px"] = np.empty(0, dtype=np.float64)
    else:
        results.update(
            pose_status=np.array("estimated"),
            T_C1C0=T_C1C0,
            inlier_indices=inlier_indices,
            inlier_reprojection_errors_px=errors,
        )
        print(f"RANSAC inliers: {len(inlier_indices)}")
        print(
            "Inlier ratio (inliers / correspondences): "
            f"{len(inlier_indices) / len(points_C0):.4f}"
        )
        print("Estimated T_C1C0 (C0 -> C1, translation in m):")
        print(np.array2string(T_C1C0, precision=8))
        print(
            "Inlier reprojection errors (px): "
            f"mean={errors.mean():.4f}, median={np.median(errors):.4f}, "
            f"RMSE={np.sqrt(np.mean(errors**2)):.4f}, max={errors.max():.4f}"
        )
        print(f"Estimated translation (m): {T_C1C0[:3, 3]}")
        print(
            f"Estimated translation magnitude (m): {np.linalg.norm(T_C1C0[:3, 3]):.6f}"
        )

        # ScanNet's registered C_D and C_R share the optical frame in each view.
        T_WC0, T_WC1 = frame_0.T_WC_D, frame_1.T_WC_D
        if T_WC0 is None or T_WC1 is None:
            print("GT comparison unavailable: a reference pose is missing")
        else:
            try:
                T_C1C0_gt = np.linalg.inv(T_WC1) @ T_WC0
            except np.linalg.LinAlgError:
                print(
                    f"GT comparison unavailable: frame-{args.frame1} "
                    "reference pose is singular"
                )
            else:
                results["T_C1C0_gt"] = T_C1C0_gt
                translation_error = np.linalg.norm(T_C1C0[:3, 3] - T_C1C0_gt[:3, 3])
                print("GT T_C1C0 = inv(T_WC1) @ T_WC0:")
                print(np.array2string(T_C1C0_gt, precision=8))
                print(f"GT translation (m): {T_C1C0_gt[:3, 3]}")
                print(
                    "GT translation magnitude (m): "
                    f"{np.linalg.norm(T_C1C0_gt[:3, 3]):.6f}"
                )
                print(f"Translation-vector error norm (m): {translation_error:.6f}")

                # Local diagnostic only. Preserve the raw GT without SO(3) repair.
                R_gt = T_C1C0_gt[:3, :3]
                R_est = T_C1C0[:3, :3]
                print(
                    "Rotation matrix difference (Frobenius norm, raw GT): "
                    f"{np.linalg.norm(R_est - R_gt):.6f}"
                )
                rotation_atol = 1e-4
                if np.allclose(
                    R_gt.T @ R_gt, np.eye(3), rtol=0, atol=rotation_atol
                ) and np.isclose(np.linalg.det(R_gt), 1, rtol=0, atol=rotation_atol):
                    cosine = (np.trace(R_est @ R_gt.T) - 1) / 2
                    angle_deg = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
                    print(
                        "Rotation error (deg, raw GT accepted at atol=1e-4): "
                        f"{angle_deg:.6f}"
                    )
                else:
                    print(
                        "Rotation angle unavailable: raw GT relative rotation fails "
                        "orthonormality or determinant +1 check at atol=1e-4"
                    )

        if args.diagnose_gt_reprojection:
            inlier_points = points_C0[inlier_indices]
            inlier_pixels = pixels_1[inlier_indices]
            print_reprojection_diagnostics(
                "Estimated", inlier_points, inlier_pixels, calibration.K_R, T_C1C0
            )
            if "T_C1C0_gt" in results:
                print_reprojection_diagnostics(
                    "GT",
                    inlier_points,
                    inlier_pixels,
                    calibration.K_R,
                    results["T_C1C0_gt"],
                )
            else:
                print("GT reprojection diagnostic unavailable: no relative GT pose")

        if args.diagnose_pnp_refinement:
            diagnose_pnp_refinement(
                points_C0[inlier_indices],
                pixels_1[inlier_indices],
                calibration.K_R,
                T_C1C0,
                results.get("T_C1C0_gt"),
            )

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{sequence_id}_frames{args.frame0}_{args.frame1}_sift"
    coordinates_path = output_dir / f"{stem}.npz"
    np.savez(coordinates_path, **results)

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
    print(f"Depth-valid RGB pixels (x, y): {pixels_0.shape}, {pixels_1.shape}")
    print(f"Aligned 3D-2D correspondences: {points_C0.shape}, {pixels_1.shape}")
    print(f"Coordinates: {coordinates_path}")
    print(
        f"Visualization (frame {args.frame0} left, frame {args.frame1} right): "
        f"{visualization_path}"
    )


if __name__ == "__main__":
    main()
