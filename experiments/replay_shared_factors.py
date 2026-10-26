"""Frozen-evidence diagnostics for shared identities and native depth models.

This is retrospective optimization of saved raw factors, never a causal SLAM
result. References score frozen outputs only. Changing identity classes can
change component populations, which are reported rather than silently aligned.
"""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np

from scene_recall.geometry.camera import transform_points
from scene_recall.multiview.batch_ba import BatchConfig, compact_linearize
from scene_recall.multiview.fixed_landmarks import DirectFactors, direct_cost
from scene_recall.multiview.joint_ba import landmark_layout
from scene_recall.odometry.io import load_trajectory
from scene_recall.run_metadata import code_hash, peak_rss, sha256
from scene_recall.slam.evaluation import score_components
from scene_recall.slam.shared_linearization import (
    ObservationCache,
    RelinearizationConfig,
)
from scene_recall.slam.shared_solver import solve_shared


def replay(
    run,
    output,
    *,
    identity="live",
    depth_model="native",
    iterations=32,
    preconditioner="jacobi",
    selective=False,
    gauge_transport=False,
):
    if output.exists():
        raise ValueError("choose a fresh replay directory")
    report, rows = load_trajectory(run)
    source = run / "shared_estimation.npz"
    arrays = dict(np.load(source))
    config = BatchConfig(**report["online_config"]["shared"]["solver"])
    config = replace(config, depth_model=depth_model, max_iterations=iterations)
    protocol = {
        "scope": "retrospective frozen raw-factor diagnostic, no tracking or future-state feedback",
        "source_sha256": sha256(source),
        "source_report_sha256": sha256(run / "report.json"),
        "package_sha256": code_hash(),
        "driver_sha256": sha256(Path(__file__)),
        "identity": identity,
        "config": asdict(config),
        "preconditioner": preconditioner,
        "selective": selective,
        "gauge_transport": gauge_transport,
    }
    output.mkdir(parents=True)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    identities = dict(arrays["live_identities"])
    tracks = np.array(
        [
            identities[track] if identity == "live" else track
            for track in arrays["track_id"]
        ]
    )
    observers, native = {}, set()
    for frame, track, point in zip(
        arrays["view"], tracks, arrays["native_points"], strict=True
    ):
        observers.setdefault(track, set()).add(int(frame))
        if np.isfinite(point).all():
            native.add(track)
    selected = {
        track
        for track, views in observers.items()
        if len(views) >= 2 and track in native
    }
    mask = np.isin(tracks, list(selected))
    keys = sorted(map(int, np.unique(arrays["view"])))
    depth_poses = dict(zip(arrays["camera_ids"], arrays["depth_poses"], strict=True))
    exposure = {
        key: arrays["T_RcD"][np.flatnonzero(arrays["view"] == key)[0]] for key in keys
    }
    poses = np.array([depth_poses[key] @ np.linalg.inv(exposure[key]) for key in keys])
    seeds = {}
    for frame, track, point in zip(
        arrays["view"], tracks, arrays["native_points"], strict=True
    ):
        if track in selected and track not in seeds and np.isfinite(point).all():
            seeds[track] = transform_points(point, depth_poses[frame])
    if identity == "live":
        seeds.update(
            {
                int(key): point
                for key, point in zip(
                    arrays["point_ids"], arrays["point_positions"], strict=True
                )
                if key in selected
            }
        )
    packet = {name: arrays[name][mask] for name in DirectFactors.__dataclass_fields__}
    packet["view"] = np.array([keys.index(int(frame)) for frame in packet["view"]])
    packet["track_id"] = tracks[mask]
    packet["points"] = np.array([seeds[track] for track in tracks[mask]])
    factors = DirectFactors(**packet)
    parents = {key: key for key in keys}

    def root(key):
        while parents[key] != key:
            key = parents[key]
        return key

    for track in selected:
        cameras = sorted(observers[track])
        for camera in cameras[1:]:
            a, b = root(cameras[0]), root(camera)
            parents[max(a, b)] = min(a, b)
    components = [root(key) for key in keys]
    fixed = [keys.index(key) for key in sorted(set(components))]
    started = perf_counter()
    revised, points, diagnostic = solve_shared(
        poses,
        factors,
        np.array(report["calibration"]["K_R"]),
        fixed,
        config,
        preconditioner=preconditioner,
        cache=ObservationCache(RelinearizationConfig(gauge_transport=gauge_transport))
        if selective
        else None,
    )
    ids, inverse, seeds = landmark_layout(factors)

    def costs(cameras, landmarks):
        residual, _, _ = compact_linearize(
            cameras,
            landmarks,
            factors,
            np.array(report["calibration"]["K_R"]),
            inverse,
            depth_model,
        )
        return {
            loss: direct_cost(residual, factors, replace(config, loss=loss))[0]
            for loss in ("linear", "cauchy")
        }

    references = [rows[key].reference_pose for key in keys]
    before_depth = np.array(
        [pose @ exposure[key] for key, pose in zip(keys, poses, strict=True)]
    )
    after_depth = np.array(
        [pose @ exposure[key] for key, pose in zip(keys, revised, strict=True)]
    )
    result = {
        "protocol": protocol,
        "solver": diagnostic,
        "solver_seconds": perf_counter() - started,
        "peak_rss_mib": peak_rss(),
        "initial_costs": costs(poses, seeds),
        "final_costs": costs(revised, points),
        "external_keyframes_before": score_components(
            before_depth, components, references
        ),
        "external_keyframes_after": score_components(
            after_depth, components, references
        ),
        "cameras": len(keys),
        "observations": len(factors.view),
        "landmarks": len(ids),
        "caveat": "native versus axial costs have different residual dimensions, identity ablations can split populations, neither defines an external surface truth",
    }
    np.savez_compressed(
        output / "state.npz",
        camera_ids=np.array(keys),
        component_ids=np.array(components),
        initial_depth_poses=before_depth,
        revised_depth_poses=after_depth,
        point_ids=ids,
        initial_points=seeds,
        revised_points=points,
    )
    (output / "report.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--identity", choices=("live", "native"), default="live")
    parser.add_argument("--depth-model", choices=("native", "axial"), default="native")
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument(
        "--preconditioner", choices=("jacobi", "coarse"), default="jacobi"
    )
    parser.add_argument("--selective", action="store_true")
    parser.add_argument("--gauge-transport", action="store_true")
    args = parser.parse_args()
    replay(
        args.run,
        args.output,
        identity=args.identity,
        depth_model=args.depth_model,
        iterations=args.iterations,
        preconditioner=args.preconditioner,
        selective=args.selective,
        gauge_transport=args.gauge_transport,
    )


if __name__ == "__main__":
    main()
