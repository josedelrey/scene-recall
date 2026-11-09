"""Finite nonlinear ambiguity of a retained group attached by one RGB ray."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from scene_recall.multiview.batch_ba import BatchConfig, compact_linearize
from scene_recall.multiview.fixed_landmarks import DirectFactors, direct_cost
from scene_recall.multiview.joint_ba import landmark_layout
from scene_recall.odometry.evaluation import error_statistics, pose_error
from scene_recall.odometry.io import load_trajectory
from scene_recall.run_metadata import code_hash, sha256


def probe(run, output, region, shifts):
    if output.exists():
        raise ValueError("choose a fresh cut probe output")
    report, rows = load_trajectory(run)
    source = run / "shared_estimation.npz"
    raw = dict(np.load(source))
    raw_identities = dict(raw["live_identities"])
    live = np.array([raw_identities[t] for t in raw["track_id"]])
    cached = dict(zip(raw["point_ids"], raw["point_positions"], strict=True))
    use = np.isin(live, list(cached))
    fit = [
        f
        for row in report["frames"]
        if row["diagnostics"].get("graph")
        for f in row["diagnostics"]["graph"].get("shared", ())
    ][-1]
    if (
        fit["status"] not in ("refined", "unchanged")
        or int(use.sum()) != fit["observations"]
    ):
        raise ValueError(
            "cut probe requires a valid saved objective with all initialized points"
        )
    views, tracks = raw["view"][use], live[use]
    keys = np.unique(views)
    frame_to_camera = {int(frame): i for i, frame in enumerate(keys)}
    inside = (views >= region[0]) & (views < region[1])
    inner_tracks = set(tracks[inside])
    boundary = inner_tracks & set(tracks[~inside])
    if len(boundary) != 1:
        raise ValueError("cut probe requires exactly one boundary landmark")
    boundary_id = next(iter(boundary))
    ray_rows = np.flatnonzero((tracks == boundary_id) & inside)
    if len(ray_rows) != 1 or raw["native_index"][use][ray_rows[0]] >= 0:
        raise ValueError(
            "cut probe requires exactly one inner RGB-only boundary observation"
        )
    arrays = {name: raw[name][use] for name in DirectFactors.__dataclass_fields__}
    arrays.update(
        view=np.array([frame_to_camera[frame] for frame in views]),
        track_id=tracks,
        points=np.array([cached[k] for k in tracks]),
    )
    factors = DirectFactors(**arrays)
    ids, inverse, points = landmark_layout(factors)
    depth = dict(zip(raw["camera_ids"], raw["depth_poses"], strict=True))
    E = np.array([factors.T_RcD[np.flatnonzero(views == frame)[0]] for frame in keys])
    poses = np.array([depth[frame] for frame in keys]) @ np.linalg.inv(E)
    cameras = (keys >= region[0]) & (keys < region[1])
    anchor_camera = factors.view[ray_rows[0]]
    direction = cached[boundary_id] - poses[anchor_camera, :3, 3]
    direction /= np.linalg.norm(direction)
    local = np.isin(ids, list(inner_tracks - boundary))
    config = BatchConfig(**report["online_config"]["shared"]["solver"])
    K = np.array(report["calibration"]["K_R"])
    base, _, _ = compact_linearize(
        poses, points, factors, K, inverse, config.depth_model
    )
    if base is None:
        raise ValueError("invalid original projections")
    cost, _ = direct_cost(base, factors, config)
    atlas = np.load(run / "atlas.npz")
    fixed_alignment = atlas["T_component_depth"][0] @ np.linalg.inv(
        rows[0].reference_pose
    )
    trials = []
    for shift in shifts:
        next_poses, next_points = poses.copy(), points.copy()
        next_poses[cameras, :3, 3] += shift * direction
        next_points[local] += shift * direction
        residual, _, _ = compact_linearize(
            next_poses,
            next_points,
            replace(factors, points=next_points[inverse]),
            K,
            inverse,
            config.depth_model,
        )
        if residual is None:
            raise ValueError("probe crosses the positive projection domain")
        value, _ = direct_cost(residual, factors, config)
        next_depth = next_poses @ E
        trials.append(
            {
                "shift_m": shift,
                "cost": value,
                "cost_change": value - cost,
                "max_residual_change": float(np.max(np.abs(residual - base))),
                "group_external_translation_m": error_statistics(
                    pose_error(T, fixed_alignment @ rows[frame].reference_pose)[0]
                    for frame, T in zip(keys[cameras], next_depth[cameras], strict=True)
                    if rows[frame].reference_pose is not None
                ),
            }
        )
    result = {
        "scope": "frozen finite ambiguity probe, preserved owned likelihood and camera gauge, no causal feedback or file mutation",
        "source_sha256": sha256(source),
        "package_sha256": code_hash(),
        "driver_sha256": sha256(Path(__file__)),
        "camera_frames": keys[cameras].tolist(),
        "boundary_class": int(boundary_id),
        "inside_landmarks": int(local.sum()),
        "native_samples_move_with_camera": True,
        "base_cost": cost,
        "trials": trials,
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--region", type=int, nargs=2, required=True)
    p.add_argument("--shifts", type=float, nargs="+", default=[-0.1, 0, 0.1])
    a = p.parse_args()
    probe(a.run, a.output, a.region, a.shifts)


if __name__ == "__main__":
    main()
