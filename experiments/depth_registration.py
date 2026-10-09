"""Independent depth-only projective point-to-plane ICP diagnostic.

Run from the repository root. No RGB features enter this objective. Sparse
and reconstruction poses are independent initializations and comparison poses.
The method assumes small adjacent-frame motion and organized depth images.
"""

import argparse
import hashlib
import itertools
import json
import os
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from scene_recall.datasets.scannet import ScanNetReader
from scene_recall.geometry.camera import (
    backproject_depth,
    project_points,
    reject_depth_edges,
    transform_points,
)
from scene_recall.odometry.evaluation import pose_error
from scene_recall.odometry.pipeline import is_rigid

FACTORS = (4, 2, 1)
DISTANCES_M = (0.12, 0.08, 0.05)
HUBER_M = 0.01
TARGETS = (9, 10, 11, 109, 110, 111, 189, 190, 191)


def stats(values):
    values = np.asarray(values)
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "rmse": None, "p95": None}
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "p95": float(np.quantile(values, 0.95)),
    }


def huber(residual):
    absolute = np.abs(residual)
    return np.where(
        absolute <= HUBER_M, 0.5 * residual**2, HUBER_M * (absolute - 0.5 * HUBER_M)
    )


def increment(update, scale=1.0):
    result = np.eye(4)
    result[:3, :3] = cv2.Rodrigues(np.asarray(update[3:], float))[0]
    result[:3, 3] = np.asarray(update[:3]) * scale
    return result


def make_cloud(depth, K, factor=1, normal_radius_native=3):
    """Retain original z-depth samples and use smoothed depth only for normals.

    Subsampling starts at native pixel zero, so every entry of K's first two
    rows is divided by factor. Missing depth remains unavailable. Normals use
    central differences of a bilateral normal-estimation image.
    """
    depth = reject_depth_edges(depth, 0.05)[::factor, ::factor].copy()
    depth[(depth < 0.3) | (depth > 6.0)] = np.nan
    intrinsic = K.copy()
    intrinsic[:2] /= factor
    points = backproject_depth(depth, intrinsic).astype(np.float64)
    smooth = cv2.bilateralFilter(np.nan_to_num(depth).astype(np.float32), 5, 0.02, 2)
    smooth[~np.isfinite(depth)] = np.nan
    normal_points = backproject_depth(smooth, intrinsic)
    radius = max(1, normal_radius_native // factor)
    normals = np.full(points.shape, np.nan)
    du = (
        normal_points[radius:-radius, 2 * radius :]
        - normal_points[radius:-radius, : -2 * radius]
    )
    dv = (
        normal_points[2 * radius :, radius:-radius]
        - normal_points[: -2 * radius, radius:-radius]
    )
    cross = np.cross(du, dv)
    length = np.linalg.norm(cross, axis=2, keepdims=True)
    np.divide(cross, length, out=cross, where=length > 1e-12)
    cross[length[..., 0] <= 1e-12] = np.nan
    normals[radius:-radius, radius:-radius] = cross
    valid = np.isfinite(points).all(axis=2) & np.isfinite(normals).all(axis=2)
    normals[~valid] = np.nan
    v, u = np.indices(depth.shape)
    sample = valid & (u % 2 == 0) & (v % 2 == 0)
    # Native 16x16-pixel tiles are split without referring to any pose.
    heldout = ((u * factor // 16 + v * factor // 16) % 2) == 1
    return {
        "points": points.reshape(-1, 3),
        "normals": normals.reshape(-1, 3),
        "K": intrinsic,
        "shape": depth.shape,
        "factor": factor,
        "train": np.flatnonzero(sample & ~heldout),
        "test": np.flatnonzero(sample & heldout),
        "all": np.flatnonzero(sample),
        "valid_normal_count": int(valid.sum()),
    }


def correspondences(source, target, pose, indices, distance=0.05):
    """Project each selected source point onto the nearest target pixel.

    Support requires positive visible projection, valid source/target normals,
    point distance within the gate, and normals agreeing within 45 degrees.
    Returned source indices always refer to the same source depth array.
    """
    moved = transform_points(source["points"][indices], pose)
    pixels = project_points(moved, target["K"])
    h, w = target["shape"]
    valid = (
        np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] <= w - 1)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] <= h - 1)
    )
    selected = indices[valid]
    moved = moved[valid]
    uv = np.floor(pixels[valid] + 0.5).astype(int)
    target_indices = uv[:, 1] * w + uv[:, 0]
    fixed = target["points"][target_indices]
    normal = target["normals"][target_indices]
    source_normal = source["normals"][selected] @ pose[:3, :3].T
    valid = (
        np.isfinite(fixed).all(axis=1)
        & np.isfinite(normal).all(axis=1)
        & (np.linalg.norm(moved - fixed, axis=1) <= distance)
        & (np.sum(source_normal * normal, axis=1) >= np.cos(np.pi / 4))
    )
    selected, moved, fixed, normal, target_indices = (
        x[valid] for x in (selected, moved, fixed, normal, target_indices)
    )
    residual = np.sum((moved - fixed) * normal, axis=1)
    return {
        "source_ids": selected,
        "target_ids": target_indices,
        "moved": moved,
        "fixed": fixed,
        "normals": normal,
        "residual": residual,
    }


def linearization(correspondence, scale):
    normals = correspondence["normals"]
    J = np.column_stack([normals * scale, np.cross(correspondence["moved"], normals)])
    residual = correspondence["residual"]
    weights = np.minimum(1, HUBER_M / np.maximum(np.abs(residual), 1e-12))
    H = J.T @ (weights[:, None] * J)
    gradient = J.T @ (weights * residual)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    rank = int(np.count_nonzero(eigenvalues > max(eigenvalues[-1] * 1e-8, 1e-12)))
    return J, H, gradient, eigenvalues, eigenvectors, rank


def register(source_pyramid, target_pyramid, initial, max_iterations=35):
    """Huber IRLS with frozen-correspondence line search and left pose updates."""
    if not is_rigid(initial):
        raise ValueError("registration initialization must be rigid")
    pose = initial.copy()
    trace = []
    levels = []
    for source, target, distance in zip(source_pyramid, target_pyramid, DISTANCES_M):
        if len(source["train"]) < 30:
            levels.append(
                {
                    "factor": source["factor"],
                    "status": "insufficient_support",
                    "iterations": 0,
                }
            )
            break
        scale = float(np.median(source["points"][source["train"], 2]))
        status = "iteration_limit"
        for iteration in range(max_iterations):
            c = correspondences(source, target, pose, source["train"], distance)
            if len(c["residual"]) < 30:
                status = "insufficient_support"
                break
            _, H, gradient, eigenvalues, _, rank = linearization(c, scale)
            if rank < 6:
                status = "rank_deficient"
                break
            update = -np.linalg.solve(H + np.eye(6) * eigenvalues[-1] * 1e-8, gradient)
            before = float(huber(c["residual"]).mean())
            accepted = False
            for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625):
                delta = increment(update * alpha, scale)
                moved = transform_points(c["moved"], delta)
                residual = np.sum((moved - c["fixed"]) * c["normals"], axis=1)
                after = float(huber(residual).mean())
                if after <= before + 1e-15:
                    pose = delta @ pose
                    accepted = True
                    break
            step_translation = float(np.linalg.norm(update[:3] * scale) * alpha)
            step_rotation = float(np.degrees(np.linalg.norm(update[3:])) * alpha)
            trace.append(
                {
                    "factor": source["factor"],
                    "iteration": iteration,
                    "support": len(c["residual"]),
                    "rank": rank,
                    "frozen_huber_before_m2": before,
                    "frozen_huber_after_m2": after,
                    "step_translation_m": step_translation,
                    "step_rotation_deg": step_rotation,
                    "accepted": accepted,
                }
            )
            if not accepted:
                status = "line_search_stalled"
                break
            if step_translation < 1e-5 and step_rotation < 0.001:
                status = "step_tolerance"
                break
        levels.append(
            {"factor": source["factor"], "status": status, "iterations": iteration + 1}
        )
        if status in ("insufficient_support", "rank_deficient"):
            break
    return {"pose": pose, "levels": levels, "trace": trace}


def score(correspondence, total):
    residual = correspondence["residual"]
    return {
        "support": len(residual),
        "support_fraction": len(residual) / total if total else 0,
        "signed_point_to_plane_mm": stats(residual * 1000),
        "absolute_point_to_plane_mm": stats(abs(residual) * 1000),
        "point_distance_mm": stats(
            np.linalg.norm(correspondence["moved"] - correspondence["fixed"], axis=1)
            * 1000
        ),
        "huber_mean_m2": float(huber(residual).mean()) if len(residual) else None,
    }


def compare_models(source, target, poses, population="test"):
    indices = source[population]
    rows = {
        name: correspondences(source, target, T, indices) for name, T in poses.items()
    }
    common = indices
    for c in rows.values():
        common = np.intersect1d(common, c["source_ids"])
    result = {}
    for name, c in rows.items():
        mask = np.isin(c["source_ids"], common)
        shared = {key: value[mask] for key, value in c.items()}
        result[name] = {
            "own_support": score(c, len(indices)),
            "common_support": score(shared, len(indices)),
        }
    return result, common


def observability(source, target, pose):
    c = correspondences(source, target, pose, source["train"])
    scale = float(np.median(source["points"][source["train"], 2]))
    _, H, gradient, eigenvalues, eigenvectors, rank = linearization(c, scale)
    normals = c["normals"]
    normal_eigenvalues = np.linalg.eigvalsh(normals.T @ normals / len(normals))
    result = {
        "rank": rank,
        "translation_scale_m": scale,
        "jacobian_condition": float(np.sqrt(eigenvalues[-1] / eigenvalues[0]))
        if rank == 6
        else None,
        "hessian_eigenvalues": eigenvalues.tolist(),
        "normal_gram_eigenvalues": normal_eigenvalues.tolist(),
        "weakest_mode_scaled_tx_ty_tz_rx_ry_rz": eigenvectors[:, 0].tolist(),
    }
    if rank == 6:
        next_update = -np.linalg.solve(H, gradient)
        result["linearized_remaining_step_m_deg"] = [
            float(np.linalg.norm(next_update[:3] * scale)),
            float(np.degrees(np.linalg.norm(next_update[3:]))),
        ]
    return result


def weak_profile(source, target, pose, obs):
    mode = np.array(obs["weakest_mode_scaled_tx_ty_tz_rx_ry_rz"])
    amounts = np.linspace(-0.02, 0.02, 41)
    poses = {
        str(i): increment(mode * a, obs["translation_scale_m"]) @ pose
        for i, a in enumerate(amounts)
    }
    scores, common = compare_models(source, target, poses)
    return {
        "amounts_scaled": amounts.tolist(),
        "common_support_count": len(common),
        "common_huber_mean_m2": [
            scores[str(i)]["common_support"]["huber_mean_m2"]
            for i in range(len(amounts))
        ],
        "common_rmse_mm": [
            scores[str(i)]["common_support"]["signed_point_to_plane_mm"]["rmse"]
            for i in range(len(amounts))
        ],
    }


def residual_image(source, target, poses, output):
    """Shared held-out source points colored by signed plane distance."""
    _, common = compare_models(source, target, poses)
    h, w = source["shape"]
    canvas = Image.new("RGB", (w * len(poses), h + 55), "white")
    draw = ImageDraw.Draw(canvas)
    for column, (name, T) in enumerate(poses.items()):
        c = correspondences(source, target, T, common)
        z = source["points"][:, 2].reshape(h, w)
        grey = np.nan_to_num(np.clip((z - 0.3) / 3 * 180 + 30, 0, 210)).astype(np.uint8)
        rgb = np.repeat(grey[..., None], 3, axis=2)
        for idx, r in zip(c["source_ids"], c["residual"]):
            y, x = divmod(int(idx), w)
            amount = min(abs(r) / 0.02, 1)
            color = (
                np.array([255, 255 * (1 - amount), 255 * (1 - amount)])
                if r > 0
                else np.array([255 * (1 - amount), 255 * (1 - amount), 255])
            )
            rgb[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2] = color.astype(np.uint8)
        canvas.paste(Image.fromarray(rgb), (column * w, 55))
        draw.text((column * w + 10, 8), name, fill="black", font_size=22)
        draw.text(
            (column * w + 10, 32),
            "signed plane residual, blue -20 / red +20 mm",
            fill="black",
            font_size=13,
        )
    canvas.save(output)


def load_runs(run_dirs):
    """Verify archived indexing, forward transforms, and reference conventions."""
    estimates = {}
    references = {}
    summaries = []
    for run in run_dirs:
        report = json.loads((run / "report.json").read_text())
        archive = np.load(run / "trajectory.npz", allow_pickle=False)
        start = report["source"]["start"]
        np.testing.assert_array_equal(
            archive["source_frame_index"], archive["frame_index"] + start
        )
        np.testing.assert_array_equal(
            archive["frame_index"], np.arange(len(archive["frame_index"]))
        )
        errors = []
        accumulation_max = 0.0
        for index, source_index in enumerate(archive["source_frame_index"]):
            if int(source_index) in references:
                np.testing.assert_array_equal(
                    references[int(source_index)], archive["T_WC_D_reference"][index]
                )
            references[int(source_index)] = archive["T_WC_D_reference"][index]
            if index == 0 or archive["status"][index] != "tracked":
                continue
            T = archive["T_C1C0"][index]
            accumulated = (
                np.linalg.inv(archive["T_SC_D"][index]) @ archive["T_SC_D"][index - 1]
            )
            difference = float(np.max(abs(T - accumulated)))
            accumulation_max = max(accumulation_max, difference)
            if difference > 1e-10:
                raise ValueError(
                    "stored pair pose disagrees with trajectory convention"
                )
            estimates[int(source_index)] = T
            gt = (
                np.linalg.inv(archive["T_WC_D_reference"][index])
                @ archive["T_WC_D_reference"][index - 1]
            )
            if is_rigid(gt):
                errors.append([int(source_index), *pose_error(T, gt)])
        phase = {}
        for k in range(10):
            phase[str(k)] = {
                "translation_mm": stats(
                    [e[1] * 1000 for e in errors if e[0] % 10 == k]
                ),
                "rotation_deg": stats([e[2] for e in errors if e[0] % 10 == k]),
            }
        boundary = [e for e in errors if e[0] % 10 == 0]
        local_contrasts = []
        lookup = {e[0]: e for e in errors}
        for e in boundary:
            if e[0] - 1 in lookup and e[0] + 1 in lookup:
                average = (np.array(lookup[e[0] - 1][1:]) + lookup[e[0] + 1][1:]) / 2
                local_contrasts.append((np.array(e[1:]) - average) * [1000, 1])
        contrasts = np.array(local_contrasts).reshape(-1, 2)
        summaries.append(
            {
                "run": str(run),
                "source": report["source"],
                "config": report["config"],
                "report_sha256": hashlib.sha256(
                    (run / "report.json").read_bytes()
                ).hexdigest(),
                "archive_sha256": hashlib.sha256(
                    (run / "trajectory.npz").read_bytes()
                ).hexdigest(),
                "pair_vs_accumulation_max_abs_difference": accumulation_max,
                "by_absolute_target_mod_10": phase,
                "errors_target_mm_deg": [[e[0], e[1] * 1000, e[2]] for e in errors],
                "boundary_minus_neighbor_average_translation_mm": stats(
                    contrasts[:, 0]
                ),
                "boundary_minus_neighbor_average_rotation_deg": stats(contrasts[:, 1]),
                "boundary_larger_than_neighbors_fraction": float(
                    np.mean(contrasts[:, 1] > 0)
                )
                if len(contrasts)
                else None,
            }
        )
    return estimates, references, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sens-path", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--targets", type=int, nargs="+", default=TARGETS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-iterations", type=int, default=35)
    parser.add_argument("--perturbations", type=int, default=6)
    args = parser.parse_args()
    if args.max_iterations < 1 or args.perturbations < 0 or min(args.targets) < 1:
        parser.error(
            "require positive iterations and target indices, nonnegative perturbations"
        )
    if args.output_dir.exists():
        parser.error("output directory exists, choose a new directory")
    args.output_dir.mkdir(parents=True)
    cv2.setNumThreads(1)
    estimates, references, periods = load_runs(args.runs)
    source_stat = args.sens_path.stat()
    for period in periods:
        identity = period["source"]
        if (
            Path(identity["path"]).resolve() != args.sens_path.resolve()
            or identity["size_bytes"] != source_stat.st_size
            or identity["mtime_ns"] != source_stat.st_mtime_ns
        ):
            parser.error("archives and depth source must identify the same capture")
    needed = set(args.targets) | {t - 1 for t in args.targets}
    observations = {}
    groups = itertools.groupby(
        enumerate(sorted(needed)), lambda entry: entry[1] - entry[0]
    )
    for _, group in groups:
        indices = [value for _, value in group]
        with ScanNetReader(
            args.sens_path, start=indices[0], stop=indices[-1] + 1
        ) as reader:
            calibration = reader.calibration
            frame_count = reader.frame_count
            for observation in reader:
                index = observation.provenance["original_frame_index"]
                np.testing.assert_array_equal(observation.T_WC_D, references[index])
                observations[index] = observation
    results = []
    rng = np.random.default_rng(0)
    for target_index in args.targets:
        source_obs, target_obs = (
            observations[target_index - 1],
            observations[target_index],
        )
        source = [make_cloud(source_obs.depth, calibration.K_D, f) for f in FACTORS]
        target = [make_cloud(target_obs.depth, calibration.K_D, f) for f in FACTORS]
        gt = np.linalg.inv(target_obs.T_WC_D) @ source_obs.T_WC_D
        vo = estimates[target_index]
        fitted = {}
        archive = {
            "T_vo": vo,
            "T_reference": gt,
            "K_D": calibration.K_D,
            "depth_source_m": source_obs.depth,
            "depth_target_m": target_obs.depth,
            "source_test_ids": source[-1]["test"],
        }
        for direction, s, t in (
            ("forward", source, target),
            ("backward", target, source),
        ):
            fitted[direction] = {}
            for name, pose in (("vo", vo), ("reference", gt)):
                init = pose if direction == "forward" else np.linalg.inv(pose)
                fit = register(s, t, init, args.max_iterations)
                archive[f"T_{direction}_{name}_init"] = fit["pose"]
                fitted[direction][name] = fit
        forward_poses = {
            "VO": vo,
            "reference": gt,
            "ICP from VO": fitted["forward"]["vo"]["pose"],
            "ICP from reference": fitted["forward"]["reference"]["pose"],
        }
        comparison, common = compare_models(source[-1], target[-1], forward_poses)
        reverse_comparison, reverse_common = compare_models(
            target[-1],
            source[-1],
            {
                "VO": np.linalg.inv(vo),
                "reference": np.linalg.inv(gt),
                "ICP from VO": fitted["backward"]["vo"]["pose"],
                "ICP from reference": fitted["backward"]["reference"]["pose"],
            },
        )
        archive["common_source_test_ids"] = common
        row = {
            "source": target_index - 1,
            "target": target_index,
            "sparse_error_to_reference_m_deg": pose_error(vo, gt),
            "source_test_count": len(source[-1]["test"]),
            "target_test_count": len(target[-1]["test"]),
            "common_forward_count": len(common),
            "common_backward_count": len(reverse_common),
            "forward_scores": comparison,
            "backward_scores": reverse_comparison,
            "registrations": {},
            "initialization_disagreement_m_deg": pose_error(
                fitted["forward"]["vo"]["pose"], fitted["forward"]["reference"]["pose"]
            ),
        }
        for direction, direction_fits in fitted.items():
            row["registrations"][direction] = {}
            for name, fit in direction_fits.items():
                forward = (
                    fit["pose"]
                    if direction == "forward"
                    else np.linalg.inv(fit["pose"])
                )
                row["registrations"][direction][name] = {
                    "levels": fit["levels"],
                    "trace": fit["trace"],
                    "pose_forward": forward.tolist(),
                    "error_to_reference_m_deg": pose_error(forward, gt),
                    "error_to_vo_m_deg": pose_error(forward, vo),
                    "forward_backward_cycle_m_deg": pose_error(
                        fitted["backward"][name]["pose"]
                        @ fitted["forward"][name]["pose"],
                        np.eye(4),
                    ),
                }
        obs = observability(source[-1], target[-1], forward_poses["ICP from VO"])
        row["observability"] = obs
        if target_index in (10, 110, 190):
            sensitivity = []
            initializations = [np.eye(4)]
            for _ in range(args.perturbations):
                update = np.r_[rng.normal(0, 0.02, 3), rng.normal(0, np.radians(1), 3)]
                initializations.append(increment(update) @ vo)
            for index, init in enumerate(initializations):
                fit = register(source, target, init, args.max_iterations)
                sensitivity.append(
                    {
                        "initial_forward": init.tolist(),
                        "levels": fit["levels"],
                        "difference_from_primary_m_deg": pose_error(
                            fit["pose"], forward_poses["ICP from VO"]
                        ),
                        "error_to_reference_m_deg": pose_error(fit["pose"], gt),
                    }
                )
                archive[f"T_sensitivity_{index}"] = fit["pose"]
            row["initialization_sensitivity"] = sensitivity
            variants = []
            for radius in (5, 9):
                alternate_source = [
                    make_cloud(source_obs.depth, calibration.K_D, f, radius)
                    for f in FACTORS
                ]
                alternate_target = [
                    make_cloud(target_obs.depth, calibration.K_D, f, radius)
                    for f in FACTORS
                ]
                alternate = register(
                    alternate_source, alternate_target, vo, args.max_iterations
                )
                variants.append(
                    {
                        "normal_radius_native_px": radius,
                        "levels": alternate["levels"],
                        "difference_from_primary_m_deg": pose_error(
                            alternate["pose"], forward_poses["ICP from VO"]
                        ),
                        "error_to_reference_m_deg": pose_error(alternate["pose"], gt),
                        "observability": observability(
                            alternate_source[-1],
                            alternate_target[-1],
                            alternate["pose"],
                        ),
                    }
                )
                archive[f"T_normal_radius_{radius}"] = alternate["pose"]
            row["normal_neighborhood_sensitivity"] = variants
            row["weak_mode_heldout_profile"] = weak_profile(
                source[-1], target[-1], forward_poses["ICP from VO"], obs
            )
        residual_image(
            source[-1],
            target[-1],
            forward_poses,
            args.output_dir / f"residuals_{target_index - 1}_{target_index}.png",
        )
        np.savez_compressed(
            args.output_dir / f"pair_{target_index - 1}_{target_index}.npz", **archive
        )
        results.append(row)
        print(
            f"{target_index - 1}->{target_index}: ICP vs reference {row['registrations']['forward']['vo']['error_to_reference_m_deg']}, "
            f"init difference {row['initialization_disagreement_m_deg']}, heldout plane RMSE "
            f"{[(n, round(v['common_support']['signed_point_to_plane_mm']['rmse'], 3)) for n, v in comparison.items()]}",
            flush=True,
        )
    stat = args.sens_path.stat()
    files = [
        Path(__file__),
        Path("src/scene_recall/geometry/camera.py"),
        Path("src/scene_recall/datasets/scannet.py"),
    ]
    report = {
        "invocation": sys.argv,
        "source": {
            "path": str(args.sens_path.resolve()),
            "frame_count": frame_count,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "opencv_threads": cv2.getNumThreads(),
            "openblas_num_threads_env": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "config": {
            "factors": FACTORS,
            "distance_gates_m": DISTANCES_M,
            "huber_m": HUBER_M,
            "native_depth_edge_m": 0.05,
            "max_iterations": args.max_iterations,
            "normal_gate_deg": 45,
            "tile_split_native_px": 16,
            "sample_stride_per_level": 2,
            "depth_range_m": [0.3, 6],
            "perturbations": args.perturbations,
            "normal_bilateral_diameter": 5,
            "normal_bilateral_sigma_m": 0.02,
            "normal_bilateral_sigma_space_px": 2,
            "normal_radius": [1, 1, 3],
            "translation_tolerance_m": 1e-5,
            "rotation_tolerance_deg": 0.001,
        },
        "calibration": {
            "K_D": calibration.K_D.tolist(),
            "T_RD": calibration.T_RD.tolist(),
        },
        "code_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
        "convention": "forward source optical depth frame to target optical depth frame",
        "periodicity": periods,
        "pairs": results,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
