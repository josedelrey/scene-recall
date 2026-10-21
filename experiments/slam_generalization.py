"""Falsify frozen RGB-D objectives with disjoint landmark prediction tests.

This source-free diagnostic is offline. It never changes association, binding,
exposure transforms or production defaults. References enter scoring only.
Withheld track measurements never initialize or fit training landmarks.
"""

import argparse
import json
from collections import Counter
from dataclasses import asdict, replace
from itertools import pairwise
from pathlib import Path
from time import perf_counter

import numpy as np

from scene_recall.geometry.camera import transform_points
from scene_recall.multiview.batch_ba import BatchConfig, solve_batch
from scene_recall.multiview.fixed_landmarks import DirectFactors
from scene_recall.odometry.evaluation import pose_error
from scene_recall.odometry.io import load_trajectory
from scene_recall.run_metadata import code_hash, sha256

MODELS = ("native", "axial", "weak_depth", "first_depth")


def subset(factors, mask):
    return replace(
        factors,
        **{
            name: getattr(factors, name)[mask].copy()
            for name in factors.__dataclass_fields__
        },
    )


def summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "count": int(values.size),
        "mean": float(values.mean()) if values.size else None,
        "rmse": float(np.sqrt(np.mean(values**2))) if values.size else None,
        "median": float(np.median(values)) if values.size else None,
        "p95": float(np.percentile(values, 95)) if values.size else None,
    }


def camera_components(factors, count):
    """Combinatorial connectivity is necessary, without implying geometric rank."""
    neighbors = [set() for _ in range(count)]
    for track in np.unique(factors.track_id):
        views = set(map(int, factors.view[factors.track_id == track]))
        for view in views:
            neighbors[view].update(views)
    unseen, components = set(range(count)), []
    while unseen:
        pending, component = [min(unseen)], set()
        while pending:
            view = pending.pop()
            if view in component:
                continue
            component.add(view)
            pending.extend(neighbors[view] - component)
        unseen -= component
        components.append(sorted(component))
    return components


def seed_from_first_depth(factors, poses):
    """Initialize each point using only its earliest valid native observation."""
    points = np.empty_like(factors.points)
    for track in np.unique(factors.track_id):
        rows = np.flatnonzero(factors.track_id == track)
        valid = rows[np.isfinite(factors.native_points[rows]).all(axis=1)]
        if not len(valid):
            raise ValueError("track has no metric seed")
        first = min(valid, key=lambda i: (factors.view[i], i))
        point = transform_points(
            factors.native_points[first],
            poses[factors.view[first]] @ factors.T_RcD[first],
        )
        points[rows] = point
    return replace(factors, points=points)


def model_factors(factors, model):
    if model not in MODELS:
        raise ValueError("unknown model")
    if model == "weak_depth":
        return replace(factors, whitening_D=factors.whitening_D / 4)
    if model == "first_depth":
        native = factors.native_points.copy()
        for track in np.unique(factors.track_id):
            rows = np.flatnonzero(factors.track_id == track)
            valid = rows[np.isfinite(native[rows]).all(axis=1)]
            if len(valid):
                first = min(valid, key=lambda i: (factors.view[i], i))
                native[rows[rows != first]] = np.nan
        return replace(factors, native_points=native)
    return factors


def split_tracks(factors, fold=0):
    """Stable identity split, before any residual or reference inspection."""
    if fold not in range(4):
        raise ValueError("holdout fold must be in range 0 to 3")
    test = factors.track_id % 4 == fold
    return subset(factors, ~test), subset(factors, test)


def prediction_errors(poses, factors, K):
    """Predict later observations from one withheld native sample per track.

    All withheld identities stay outside pose fitting. The first observation
    conditions a test point and is excluded from scoring. This measures
    cross-view consistency and includes seed noise, without surface-truth claims.
    """
    rgb, axial, native, invalid = [], [], [], 0
    track_rgb, track_axial = [], []
    for track in np.unique(factors.track_id):
        rows = np.flatnonzero(factors.track_id == track)
        valid = rows[np.isfinite(factors.native_points[rows]).all(axis=1)]
        if not len(valid):
            continue
        first = min(valid, key=lambda i: (factors.view[i], i))
        point = transform_points(
            factors.native_points[first],
            poses[factors.view[first]] @ factors.T_RcD[first],
        )
        pr, pz = [], []
        for row in rows[factors.view[rows] > factors.view[first]]:
            q = transform_points(point, np.linalg.inv(poses[factors.view[row]]))
            if q[2] <= 0.05:
                invalid += 1
                pixel_error = 10.0
            else:
                projected = K @ q
                pixel_error = float(
                    np.linalg.norm(projected[:2] / projected[2] - factors.pixels[row])
                )
            rgb.append(pixel_error)
            pr.append(min(pixel_error, 10.0))
            if np.isfinite(factors.native_points[row]).all():
                qd = transform_points(q, np.linalg.inv(factors.T_RcD[row]))
                error = qd - factors.native_points[row]
                axial.append(abs(float(error[2])))
                native.append(float(np.linalg.norm(error)))
                pz.append(min(abs(float(error[2])), 0.1))
        if pr:
            track_rgb.append(float(np.mean(pr)))
        if pz:
            track_axial.append(float(np.mean(pz)))
    return {
        "rgb_px": summary(rgb),
        "axial_m": summary(axial),
        "native_m": summary(native),
        "track_balanced_capped_rgb_px": summary(track_rgb),
        "track_balanced_capped_axial_m": summary(track_axial),
        "invalid_projections": invalid,
    }


def reference_errors(poses, references):
    valid = [i for i, reference in enumerate(references) if reference is not None]
    if len(valid) < 2:
        return {"available": False}
    first = valid[0]
    transform = poses[first] @ np.linalg.inv(references[first])
    anchored = [pose_error(poses[i], transform @ references[i]) for i in valid[1:]]
    relative = [
        pose_error(
            np.linalg.inv(poses[a]) @ poses[b],
            np.linalg.inv(references[a]) @ references[b],
        )
        for a, b in pairwise(valid)
    ]
    return {
        "available": True,
        "anchored_translation_m": summary([p[0] for p in anchored]),
        "anchored_rotation_deg": summary([p[1] for p in anchored]),
        "relative_translation_m": summary([p[0] for p in relative]),
        "relative_rotation_deg": summary([p[1] for p in relative]),
    }


def evaluate_block(
    poses, factors, K, references, config, fit_function=solve_batch, holdout_fold=0
):
    factors = seed_from_first_depth(factors, poses)
    train, test = split_tracks(factors, holdout_fold)
    result = {
        "train_tracks": len(np.unique(train.track_id)),
        "test_tracks": len(np.unique(test.track_id)),
        "components": camera_components(train, len(poses)),
        "initial": {
            "prediction": prediction_errors(poses, test, K),
            "reference": reference_errors(poses, references),
        },
        "models": {},
    }
    if all(p is not None for p in references):
        # A diagnostic control, never passed to a solver or used for selection.
        result["reference_pose_prediction"] = prediction_errors(
            np.array(references), test, K
        )
    for model in MODELS:
        modified = model_factors(train, model)
        started = perf_counter()
        fitted, _, diagnostic = fit_function(
            poses,
            modified,
            K,
            replace(config, depth_model="axial" if model == "axial" else "native"),
        )
        result["models"][model] = {
            "solver": diagnostic,
            "seconds": perf_counter() - started,
            "prediction": prediction_errors(fitted, test, K),
            "reference": reference_errors(fitted, references),
        }
    return result


def run_case(
    run, initial_run, config, block_size, fit_function=solve_batch, holdout_fold=0
):
    report, rows = load_trajectory(run)
    initial_report, initial_rows = load_trajectory(initial_run)
    if (
        report["source"] != initial_report["source"]
        or report["calibration"] != initial_report["calibration"]
    ):
        raise ValueError(
            "initial trajectory must share canonical source and calibration"
        )
    if len(rows) != len(initial_rows) or any(
        a.timestamp != b.timestamp or a.segment_id != b.segment_id
        for a, b in zip(rows, initial_rows, strict=True)
    ):
        raise ValueError("initial trajectory must share inputs and segment boundaries")
    with np.load(run / "measurements.npz", allow_pickle=False) as archive:
        arrays = {
            key: archive[key].copy()
            for key in DirectFactors.__dataclass_fields__
            if key in archive
        }
    result = {
        "inputs": {
            str(p): sha256(p)
            for p in (
                run / "report.json",
                run / "trajectory.npz",
                run / "measurements.npz",
                initial_run / "trajectory.npz",
            )
        },
        "blocks": [],
        "excluded": [],
    }
    if not arrays:
        result["excluded"].append({"reason": "no admitted metric factors"})
        return result
    keys = sorted(set(map(int, arrays["view"])))
    K = np.array(report["calibration"]["K_R"])
    for segment in sorted({r.segment_id for r in rows}):
        cameras = [k for k in keys if rows[k].segment_id == segment]
        for start in range(0, len(cameras), block_size):
            block = cameras[start : start + block_size]
            if len(block) != block_size:
                result["excluded"].append(
                    {"cameras": block, "reason": "incomplete final block"}
                )
                continue
            mask = np.isin(arrays["view"], block)
            packet = DirectFactors(**{k: v[mask].copy() for k, v in arrays.items()})
            ids, counts = np.unique(packet.track_id, return_counts=True)
            seeded_ids = np.unique(
                packet.track_id[np.isfinite(packet.native_points).all(axis=1)]
            )
            mask = np.isin(
                packet.track_id, np.intersect1d(ids[counts >= 2], seeded_ids)
            )
            removed = int((~mask).sum())
            if not mask.any():
                result["excluded"].append(
                    {"cameras": block, "reason": "no shared tracks"}
                )
                continue
            packet = subset(packet, mask)
            packet = replace(
                packet, view=np.array([block.index(int(v)) for v in packet.view])
            )
            transforms = {
                i: packet.T_RcD[np.flatnonzero(packet.view == i)[0]]
                for i in np.unique(packet.view)
            }
            if (
                len(transforms) != len(block)
                or not (packet.track_id % 4 == holdout_fold).any()
                or not (packet.track_id % 4 != holdout_fold).any()
            ):
                result["excluded"].append(
                    {"cameras": block, "reason": "camera or split lacks support"}
                )
                continue
            poses = np.array(
                [
                    initial_rows[key].T_SC_D @ np.linalg.inv(transforms[i])
                    for i, key in enumerate(block)
                ]
            )
            references = [
                None
                if rows[key].reference_pose is None
                else rows[key].reference_pose @ np.linalg.inv(transforms[i])
                for i, key in enumerate(block)
            ]
            record = evaluate_block(
                poses, packet, K, references, config, fit_function, holdout_fold
            )
            record.update(
                cameras=block,
                segment=segment,
                factor_sha256=packet.fingerprint(),
                excluded_unshared_or_unseeded_observations=removed,
            )
            result["blocks"].append(record)
    return result


def aggregate(cases):
    result = {}
    for name, case in cases.items():
        values = {}
        for model in ("initial", *MODELS):
            records = [
                b["initial"] if model == "initial" else b["models"][model]
                for b in case["blocks"]
            ]
            values[model] = {
                "blocks": len(records),
                "statuses": dict(
                    Counter(
                        r.get("solver", {}).get("status", "initial") for r in records
                    )
                ),
                "median_seconds": summary(
                    [r["seconds"] for r in records if "seconds" in r]
                )["median"],
                "mean_anchored_translation_rmse_m": summary(
                    [
                        r["reference"]["anchored_translation_m"]["rmse"]
                        for r in records
                        if r["reference"]["available"]
                    ]
                )["mean"],
                "mean_anchored_rotation_rmse_deg": summary(
                    [
                        r["reference"]["anchored_rotation_deg"]["rmse"]
                        for r in records
                        if r["reference"]["available"]
                    ]
                )["mean"],
                "mean_test_rgb_px": summary(
                    [
                        r["prediction"]["track_balanced_capped_rgb_px"]["mean"]
                        for r in records
                        if r["prediction"]["track_balanced_capped_rgb_px"]["count"]
                    ]
                )["mean"],
                "mean_test_axial_m": summary(
                    [
                        r["prediction"]["track_balanced_capped_axial_m"]["mean"]
                        for r in records
                        if r["prediction"]["track_balanced_capped_axial_m"]["count"]
                    ]
                )["mean"],
            }
        result[name] = values
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cases", type=Path, help="JSON mapping case names to run and initial_run paths"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--solver", choices=("joint", "components"), default="joint")
    parser.add_argument("--holdout-fold", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    if args.output_dir.exists() or args.block_size < 2:
        parser.error("choose a fresh output directory and block size >= 2")
    manifest = json.loads(args.cases.read_text())
    fit_function = solve_batch
    solver_path = Path(__file__).parent / "slam_component_replay.py"
    if args.solver == "components":
        from slam_component_replay import solve_components

        fit_function = solve_components
    config = BatchConfig(
        max_iterations=20,
        max_translation_m=0.1,
        max_rotation_deg=5,
        max_point_displacement_m=0.1,
    )
    args.output_dir.mkdir(parents=True)
    protocol = {
        "phase": "frozen before results",
        "cases": manifest,
        "cases_sha256": sha256(args.cases),
        "driver_sha256": sha256(Path(__file__)),
        "package_sha256": code_hash(),
        "config": asdict(config),
        "block_size": args.block_size,
        "models": list(MODELS),
        "solver": args.solver,
        "component_driver_sha256": sha256(solver_path),
        "split": "track_id modulo 4 equals holdout_fold held out, others train",
        "holdout_fold": args.holdout_fold,
        "selection": "disjoint complete camera blocks within source segments, shared tracks only",
        "interpretation": "offline frozen-association diagnosis, no online or independent-surface claim",
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n"
    )
    cases = {}
    for name, paths in manifest.items():
        case = run_case(
            Path(paths["run"]),
            Path(paths["initial_run"]),
            config,
            args.block_size,
            fit_function,
            args.holdout_fold,
        )
        cases[name] = case
        (args.output_dir / f"{name}.json").write_text(
            json.dumps(case, indent=2, allow_nan=False) + "\n"
        )
        print(name, len(case["blocks"]), "blocks", flush=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(aggregate(cases), indent=2, allow_nan=False) + "\n"
    )
    if (
        protocol["driver_sha256"] != sha256(Path(__file__))
        or protocol["package_sha256"] != code_hash()
        or protocol["component_driver_sha256"] != sha256(solver_path)
    ):
        raise RuntimeError("source changed during evaluation")


if __name__ == "__main__":
    main()
