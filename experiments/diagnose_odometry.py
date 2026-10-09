"""Compare unchanged sparse estimates and ScanNet references on shared data.

Run with the repository environment. This diagnostic intercepts the solver's
inputs without changing them and archives every correspondence and mask.
Images and numerical results are written to a new ignored output directory.
"""

import argparse
import hashlib
import itertools
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageDraw

from scene_recall.datasets.scannet import ScanNetReader
from scene_recall.geometry.camera import (
    backproject_rgb_pixels,
    project_points,
    reject_depth_edges,
    transform_points,
)
from scene_recall.odometry import sparse
from scene_recall.odometry.evaluation import pose_error

DEFAULT_TARGETS = [8, 9, 10, 11, 12, 108, 109, 110, 111, 112, 188, 189, 190, 191, 192]


def summary(values):
    values = np.asarray(values)
    return (
        {
            "count": len(values),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "rmse": float(np.sqrt(np.mean(values**2))),
            "p05": float(np.quantile(values, 0.05)),
            "p95": float(np.quantile(values, 0.95)),
            "std": float(values.std()),
        }
        if len(values)
        else {"count": 0}
    )


def residuals(points, pixels, K, pose):
    return project_points(transform_points(points, pose), K) - pixels


def residual_summary(residual):
    finite = np.isfinite(residual).all(axis=1)
    residual = residual[finite]
    return {
        "nonfinite_count": int((~finite).sum()),
        "norm_px": summary(np.linalg.norm(residual, axis=1)),
        "signed_u_px": summary(residual[:, 0]),
        "signed_v_px": summary(residual[:, 1]),
    }


def motion(pose):
    return {
        "translation_mm": (pose[:3, 3] * 1000).tolist(),
        "rotvec_deg": np.degrees(cv2.Rodrigues(pose[:3, :3])[0].ravel()).tolist(),
    }


def numerical_jacobian(points, K, pose, scale):
    """Left perturbations with translation scaled by median source z-depth."""
    columns = []
    epsilon = 1e-6
    for k in range(6):
        projections = []
        for sign in [-1, 1]:
            delta = np.eye(4)
            if k < 3:
                delta[k, 3] = sign * epsilon * scale
            else:
                axis = np.zeros(3)
                axis[k - 3] = sign * epsilon
                delta[:3, :3] = cv2.Rodrigues(axis)[0]
            projections.append(
                project_points(transform_points(points, delta @ pose), K)
            )
        columns.append(((projections[1] - projections[0]) / (2 * epsilon)).ravel())
    return np.array(columns).T


def geometry(points, pixels, K, pose):
    scale = float(np.median(points[:, 2]))
    J = numerical_jacobian(points, K, pose, scale)
    _, singular, vt = np.linalg.svd(J, full_matrices=False)
    covariance = np.linalg.inv(J.T @ J)
    correlation = covariance / np.sqrt(
        np.outer(np.diag(covariance), np.diag(covariance))
    )
    xyz_singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    spatial = np.floor(pixels / [1296 / 4, 968 / 3]).astype(int)
    return {
        "source_z_m": summary(points[:, 2]),
        "xyz_centered_singular_values_m": xyz_singular.tolist(),
        "smallest_to_largest_xyz_ratio": float(xyz_singular[-1] / xyz_singular[0]),
        "occupied_4x3_cells": len(np.unique(spatial, axis=0)),
        "unique_source_pixels_rounded_0_1px": len(
            np.unique(np.round(pixels, 1), axis=0)
        ),
        "jacobian_translation_scale_m": scale,
        "jacobian_singular_values_px": singular.tolist(),
        "jacobian_condition": float(singular[0] / singular[-1]),
        "weakest_mode_scaled_tx_ty_tz_rx_ry_rz": vt[-1].tolist(),
        "unit_pixel_iid_translation_std_mm": (
            np.sqrt(np.diag(covariance))[:3] * scale * 1000
        ).tolist(),
        "unit_pixel_iid_rotation_std_deg": np.degrees(
            np.sqrt(np.diag(covariance))[3:]
        ).tolist(),
        "parameter_correlation": correlation.tolist(),
    }


def spatial_depth(points, pixels, residual):
    rows = []
    for y in range(3):
        for x in range(4):
            mask = (
                (pixels[:, 0] >= x * 1296 / 4)
                & (pixels[:, 0] < (x + 1) * 1296 / 4)
                & (pixels[:, 1] >= y * 968 / 3)
                & (pixels[:, 1] < (y + 1) * 968 / 3)
            )
            rows.append(
                {"cell_xy": [x, y], "residual": residual_summary(residual[mask])}
            )
    depths = []
    edges = [0, 1.5, 2, 2.5, 3, 4, 6, 100]
    for lo, hi in itertools.pairwise(edges):
        mask = (points[:, 2] >= lo) & (points[:, 2] < hi)
        depths.append(
            {"z_range_m": [lo, hi], "residual": residual_summary(residual[mask])}
        )
    X = np.column_stack(
        [
            np.ones(len(points)),
            (pixels[:, 0] - 648) / 648,
            (pixels[:, 1] - 484) / 484,
            1 / points[:, 2],
        ]
    )
    fit = X @ np.linalg.lstsq(X, residual, rcond=None)[0]
    r2 = 1 - np.sum((residual - fit) ** 2, axis=0) / np.sum(
        (residual - residual.mean(axis=0)) ** 2, axis=0
    )
    return {
        "grid_4x3": rows,
        "depth_bins": depths,
        "affine_xy_inverse_z_R2": r2.tolist(),
    }


def lm(points, pixels, K, pose):
    rvec, tvec = cv2.solvePnPRefineLM(
        points.copy(),
        pixels.copy(),
        K,
        None,
        cv2.Rodrigues(pose[:3, :3])[0],
        pose[:3, 3].reshape(3, 1).copy(),
    )
    result = np.eye(4)
    result[:3, :3] = cv2.Rodrigues(rvec)[0]
    result[:3, 3] = tvec.ravel()
    return result


def probes(points, pixels, source_pixels, K, pose, gt, inliers, count):
    """Diagnostic fits only, never used by production estimation or scoring."""
    p, q = points[inliers], pixels[inliers]
    rng = np.random.default_rng(0)
    unique, cluster = np.unique(
        np.round(source_pixels[inliers], 1), axis=0, return_inverse=True
    )
    bootstrap = []
    for _ in range(count):
        weights = np.bincount(
            rng.integers(0, len(unique), len(unique)), minlength=len(unique)
        )
        selected = np.repeat(np.arange(len(p)), weights[cluster])
        fit = lm(p[selected], q[selected], K, pose)
        bootstrap.append([*pose_error(fit, pose), *pose_error(fit, gt)])
    bootstrap = np.array(bootstrap)
    # Split before pose fitting, score every held-out depth-valid match.
    _, all_clusters = np.unique(np.round(source_pixels, 1), axis=0, return_inverse=True)
    cluster_folds = rng.permutation(all_clusters.max() + 1) % 5
    holdout = []
    for fold in range(5):
        test = np.flatnonzero(cluster_folds[all_clusters] == fold)
        train = np.setdiff1d(np.arange(len(points)), test)
        estimate = sparse.estimate_sparse_pose(
            points[train], pixels[train], K, sparse.SparseConfig()
        )
        if estimate.T_C1C0 is None:
            holdout.append({"failure": estimate.reason})
            continue
        est_residual = residuals(points[test], pixels[test], K, estimate.T_C1C0)
        ref_residual = residuals(points[test], pixels[test], K, gt)
        holdout.append(
            {
                "count": len(test),
                "estimate": residual_summary(est_residual),
                "reference": residual_summary(ref_residual),
                "pose_error_m_deg": pose_error(estimate.T_C1C0, gt),
                "estimate_lower_residual_fraction": float(
                    np.mean(
                        np.linalg.norm(est_residual, axis=1)
                        < np.linalg.norm(ref_residual, axis=1)
                    )
                ),
            }
        )
    gt_r = residuals(p, q, K, gt)
    # Depth can only change a reference projection along its epipolar direction.
    eps = 0.001
    derivative = (
        residuals(p * (1 + eps), q, K, gt) - residuals(p * (1 - eps), q, K, gt)
    ) / (2 * eps)
    parallel = (
        derivative
        * np.sum(gt_r * derivative, axis=1, keepdims=True)
        / np.sum(derivative**2, axis=1, keepdims=True)
    )
    scales = np.linspace(0.5, 1.5, 201)
    scale_rmse = np.array(
        [
            np.sqrt(np.mean(np.sum(residuals(p * s, q, K, gt) ** 2, axis=1)))
            for s in scales
        ]
    )
    gt_initialized = lm(p, q, K, gt)
    est_initialized = lm(p, q, K, pose)
    return {
        "cluster_bootstrap_replicates": count,
        "bootstrap_displacement_from_fitted_translation_m": summary(bootstrap[:, 0]),
        "bootstrap_displacement_from_fitted_rotation_deg": summary(bootstrap[:, 1]),
        "bootstrap_error_to_reference_translation_m": summary(bootstrap[:, 2]),
        "bootstrap_error_to_reference_rotation_deg": summary(bootstrap[:, 3]),
        "holdout_5fold_all_matches": holdout,
        "gt_initialized_lm_displacement_from_estimate_m_deg": pose_error(
            gt_initialized, pose
        ),
        "gt_initialized_lm_rmse_px": float(
            np.sqrt(np.mean(np.sum(residuals(p, q, K, gt_initialized) ** 2, axis=1)))
        ),
        "best_uniform_depth_scale_at_gt": float(scales[scale_rmse.argmin()]),
        "same_population_lm_initializations_difference_m_deg": pose_error(
            gt_initialized, est_initialized
        ),
        "best_uniform_depth_scale_gt_rmse_px": float(scale_rmse.min()),
        "gt_residual_energy_perpendicular_to_depth_direction_fraction": float(
            np.sum((gt_r - parallel) ** 2) / np.sum(gt_r**2)
        ),
    }


def symmetric_depth_check(
    points, source_pixels, pixels, target, calibration, poses, mask
):
    """Check the same matched IDs with valid filtered target depth.

    This shares the SIFT measurements and estimator selection. It is not an
    independent depth registration experiment.
    """
    depth = reject_depth_edges(target.depth, 0.05)
    target_points = backproject_rgb_pixels(
        pixels, depth, calibration.K_R, calibration.K_D
    )
    valid = mask & np.isfinite(target_points).all(axis=1)
    results = {}
    for name, pose in poses.items():
        delta = transform_points(points[valid], pose) - target_points[valid]
        reverse = residuals(
            target_points[valid],
            source_pixels[valid],
            calibration.K_R,
            np.linalg.inv(pose),
        )
        results[name] = {
            "point_distance_mm": summary(np.linalg.norm(delta, axis=1) * 1000),
            "signed_z_mm": summary(delta[:, 2] * 1000),
            "reverse_reprojection": residual_summary(reverse),
        }
    return {"count": int(valid.sum()), "models": results}


def panels(source, target, pixels, est_residual, gt_residual, mask, path):
    """Pillow contact sheet with native-pixel residual vectors magnified 6x."""
    scale = 0.5
    width, height = 648, 484
    canvas = Image.new("RGB", (2 * width, height + 80), "white")
    draw = ImageDraw.Draw(canvas)
    for column, (label, residual) in enumerate(
        [("estimate", est_residual), ("reference", gt_residual)]
    ):
        background = Image.fromarray(target.rgb).resize((width, height))
        canvas.paste(background, (column * width, 80))
        draw.text(
            (column * width + 12, 10),
            f"{source.provenance['original_frame_index']} -> "
            f"{target.provenance['original_frame_index']}: {label}",
            fill="black",
            font_size=22,
        )
        draw.text(
            (column * width + 12, 40),
            "projected - observed, vectors magnified 6x",
            fill="black",
            font_size=17,
        )
        for pixel, vector in zip(pixels[mask], residual[mask]):
            a = pixel * scale + [column * width, 80]
            b = a + vector * scale * 6
            draw.line([tuple(a), tuple(b)], fill=(255, 190, 0), width=2)
            draw.ellipse([a[0] - 1, a[1] - 1, a[0] + 1, a[1] + 1], fill=(0, 220, 255))
    canvas.save(path)


def archive_phase(run):
    report = json.loads((run / "report.json").read_text())
    result = {}
    rows = [
        p for p in report["evaluation"]["pair_errors_m_deg"] if p["final"] is not None
    ]
    for phase in range(10):
        values = np.array(
            [
                p["final"]
                for p in rows
                if (p["target_frame"] + report["source"]["start"]) % 10 == phase
            ]
        )
        result[str(phase)] = {
            "translation_m": summary(values[:, 0]),
            "rotation_deg": summary(values[:, 1]),
        }
    largest = sorted(rows, key=lambda p: p["final"][1], reverse=True)[:20]
    return {
        "source_start": report["source"]["start"],
        "report_sha256": hashlib.sha256((run / "report.json").read_bytes()).hexdigest(),
        "boundary_vs_other": {
            str(boundary): {
                "translation_m": summary(
                    [
                        p["final"][0]
                        for p in rows
                        if ((p["target_frame"] + report["source"]["start"]) % 10 == 0)
                        == boundary
                    ]
                ),
                "rotation_deg": summary(
                    [
                        p["final"][1]
                        for p in rows
                        if ((p["target_frame"] + report["source"]["start"]) % 10 == 0)
                        == boundary
                    ]
                ),
            }
            for boundary in [True, False]
        },
        "by_target_source_index_mod_10": result,
        "largest_rotation_errors": largest,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sens-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--baseline-dir", type=Path, default=Path("outputs/odometry/default_0_200")
    )
    parser.add_argument("--targets", type=int, nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--phase-runs", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory exists, choose a new directory")
    args.output_dir.mkdir(parents=True)
    cv2.setNumThreads(1)
    config = sparse.SparseConfig()
    baseline_report = json.loads((args.baseline_dir / "report.json").read_text())
    if (
        baseline_report["config"] != asdict(config)
        or baseline_report["source"]["start"] != 0
    ):
        parser.error("baseline must use default configuration and start at zero")
    original_solver = sparse.estimate_sparse_pose
    captured = {}

    def capture(points, pixels, K, config):
        captured.update(points=points.copy(), pixels=pixels.copy())
        return original_solver(points, pixels, K, config)

    baseline = np.load(args.baseline_dir / "trajectory.npz")
    results = []
    observations = {}
    needed = set(args.targets) | {t - 1 for t in args.targets}
    with ScanNetReader(args.sens_path, stop=max(needed) + 1) as reader:
        calibration = reader.calibration
        for obs in reader:
            index = obs.provenance["original_frame_index"]
            if index in needed:
                observations[index] = obs
    K, T_RD = calibration.K_R, calibration.T_RD
    contact = Image.new("RGB", (648 * 4, 514 * 3), "white")
    draw = ImageDraw.Draw(contact)
    for pos, index in enumerate([8, 9, 10, 11, 108, 109, 110, 111, 188, 189, 190, 191]):
        if index in observations:
            x, y = pos % 4 * 648, pos // 4 * 514
            contact.paste(
                Image.fromarray(observations[index].rgb).resize((648, 484)), (x, y + 30)
            )
            draw.text((x + 10, y + 4), f"RGB frame {index}", fill="black", font_size=22)
    contact.save(args.output_dir / "rgb_controls.png")
    for target_index in args.targets:
        source, target = observations[target_index - 1], observations[target_index]
        with patch.object(sparse, "estimate_sparse_pose", capture):
            pair = sparse.SparseRGBDBackend(config).estimate(
                source, target, calibration
            )
        if pair.T_C1C0 is None:
            raise RuntimeError(pair.reason)
        points, pixels = captured["points"], captured["pixels"]
        source_pixels = project_points(points, K)
        estimate = T_RD @ pair.T_C1C0 @ np.linalg.inv(T_RD)
        gt_depth = np.linalg.inv(target.T_WC_D) @ source.T_WC_D
        gt = T_RD @ gt_depth @ np.linalg.inv(T_RD)
        initial, ransac_indices, _ = sparse.ransac_pnp(
            points,
            pixels,
            K,
            iterations=config.iterations,
            confidence=config.confidence,
            threshold_px=config.reprojection_threshold_px,
            seed=config.seed,
        )
        est_r = residuals(points, pixels, K, estimate)
        gt_r = residuals(points, pixels, K, gt)
        est_mask = np.linalg.norm(est_r, axis=1) <= config.reprojection_threshold_px
        gt_mask = np.linalg.norm(gt_r, axis=1) <= config.reprojection_threshold_px
        raw_mask = np.zeros(len(points), bool)
        raw_mask[ransac_indices] = True
        masks = {
            "all_matches": np.ones(len(points), bool),
            "ransac_consensus": raw_mask,
            "estimator_inliers": est_mask,
            "reference_threshold": gt_mask,
            "intersection": est_mask & gt_mask,
            "union": est_mask | gt_mask,
            "estimator_only": est_mask & ~gt_mask,
            "reference_only": gt_mask & ~est_mask,
            "neither": ~(est_mask | gt_mask),
        }
        difference = np.max(np.abs(baseline["T_C1C0"][target_index] - pair.T_C1C0))
        if difference > 1e-12:
            raise RuntimeError(f"archived estimate mismatch: {difference}")
        np.savez_compressed(
            args.output_dir / f"pair_{target_index - 1}_{target_index}.npz",
            points_R0=points,
            source_pixels=source_pixels,
            target_pixels=pixels,
            T_estimate_R1R0=estimate,
            T_reference_R1R0=gt,
            T_initial_R1R0=initial,
            residual_estimate_px=est_r,
            residual_reference_px=gt_r,
            K_R=K,
            **masks,
        )
        result = {
            "source": target_index - 1,
            "target": target_index,
            "diagnostics": pair.diagnostics,
            "archive_pose_max_abs_difference": float(difference),
            "initial_pose_error_m_deg": pose_error(initial, gt),
            "final_pose_error_m_deg": pose_error(estimate, gt),
            "estimate_motion": motion(estimate),
            "reference_motion": motion(gt),
            "source_provenance": source.provenance,
            "target_provenance": target.provenance,
            "estimate_minus_reference_translation_mm": (
                (estimate[:3, 3] - gt[:3, 3]) * 1000
            ).tolist(),
            "populations": {
                name: {
                    "count": int(mask.sum()),
                    "estimate": residual_summary(est_r[mask]),
                    "reference": residual_summary(gt_r[mask]),
                    "depth_m": summary(points[mask, 2]),
                }
                for name, mask in masks.items()
            },
            "geometry": geometry(
                points[est_mask],
                source_pixels[est_mask],
                K,
                estimate,
            ),
            "symmetric_depth_check": symmetric_depth_check(
                points,
                source_pixels,
                pixels,
                target,
                calibration,
                {"estimate": estimate, "reference": gt},
                est_mask,
            ),
            "estimator_inlier_spatial_depth_estimate": spatial_depth(
                points[est_mask], pixels[est_mask], est_r[est_mask]
            ),
            "estimator_inlier_spatial_depth_reference": spatial_depth(
                points[est_mask], pixels[est_mask], gt_r[est_mask]
            ),
        }
        if args.bootstrap > 0:
            # Restore the solver before the diagnostic held-out fits.
            result["probes"] = probes(
                points, pixels, source_pixels, K, estimate, gt, est_mask, args.bootstrap
            )
        panels(
            source,
            target,
            pixels,
            est_r,
            gt_r,
            est_mask,
            args.output_dir / f"residuals_{target_index - 1}_{target_index}.png",
        )
        results.append(result)
        print(
            f"{target_index - 1}->{target_index}: {len(points)} matches, "
            f"{est_mask.sum()} estimate inliers, {gt_mask.sum()} reference inliers, "
            f"RMSE {result['populations']['estimator_inliers']['estimate']['norm_px']['rmse']:.2f}"
            f" / {result['populations']['estimator_inliers']['reference']['norm_px']['rmse']:.2f}",
            flush=True,
        )
    stat = args.sens_path.stat()
    paths = sorted(Path("src/scene_recall").rglob("*.py")) + [Path(__file__)]
    output = {
        "config": asdict(config),
        "targets": args.targets,
        "invocation": sys.argv,
        "source": {
            "path": str(args.sens_path.resolve()),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
        "calibration": {
            "K_R": K.tolist(),
            "K_D": calibration.K_D.tolist(),
            "T_RD": T_RD.tolist(),
            "rgb_shape": calibration.rgb_shape,
            "depth_shape": calibration.depth_shape,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "opencv_threads": cv2.getNumThreads(),
        },
        "code_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        "residual_convention": "projected minus observed in native RGB pixels, u right, v down",
        "archive_phase": {
            str(run): archive_phase(run)
            for run in [args.baseline_dir, *args.phase_runs]
        },
        "pairs": results,
    }
    (args.output_dir / "diagnostics.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
