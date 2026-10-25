"""Retained connectivity, conditional information and reference contradictions.

Reference diagnostics never modify identities, factors, weights or estimates.
Large reference discrepancies cannot distinguish association, depth binding,
calibration, timing and reference error without further independent evidence.
"""

import argparse
import json
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np

from scene_recall.geometry.camera import transform_points
from scene_recall.multiview.audit import normalized_spectrum
from scene_recall.multiview.batch_ba import (
    BatchConfig,
    compact_blocks,
    compact_linearize,
)
from scene_recall.multiview.fixed_landmarks import DirectFactors, direct_cost
from scene_recall.multiview.joint_ba import landmark_layout
from scene_recall.odometry.evaluation import error_statistics
from scene_recall.odometry.io import load_trajectory
from scene_recall.run_metadata import code_hash, sha256
from scene_recall.slam.shared_solver import SchurOperator


def diagnose(run, *, blocks_output=None, global_rank=False, region=None):
    report, rows = load_trajectory(run)
    path = run / "shared_estimation.npz"
    arrays = dict(np.load(path))
    identities = dict(arrays["live_identities"])
    raw = arrays["track_id"]
    live = np.array([identities[track] for track in raw])
    views = arrays["view"]
    keys = np.unique(views)
    native = np.isfinite(arrays["native_points"]).all(axis=1)
    reference = np.array(
        [rows[int(frame)].reference_pose is not None for frame in views]
    )
    world = np.full_like(arrays["native_points"], np.nan)
    for frame in keys:
        mask = (views == frame) & native & reference
        if mask.any():
            world[mask] = transform_points(
                arrays["native_points"][mask], rows[int(frame)].reference_pose
            )

    def identity_radii(tracks):
        samples = defaultdict(list)
        for track, point in zip(
            tracks[native & reference], world[native & reference], strict=True
        ):
            samples[int(track)].append(point)
        radii = [
            float(
                np.sqrt(
                    np.mean(np.sum((np.array(p) - np.mean(p, axis=0)) ** 2, axis=1))
                )
            )
            for p in samples.values()
            if len(p) >= 2
        ]
        return {
            "rms_radius_m": error_statistics(radii),
            "above_5cm": sum(value > 0.05 for value in radii),
            "above_10cm": sum(value > 0.1 for value in radii),
        }

    by = {frame: set(raw[views == frame]) for frame in keys}
    live_by = {frame: set(live[views == frame]) for frame in keys}
    links = [
        {
            "source": int(a),
            "target": int(b),
            "native_tracks": len(by[a] & by[b]),
            "live_classes": len(live_by[a] & live_by[b]),
        }
        for a, b in pairwise(keys)
    ]
    observers = defaultdict(list)
    depth_classes = set(live[native])
    for frame, track in zip(views, live, strict=True):
        observers[int(track)].append(int(frame))
    selected = {
        track
        for track, frames in observers.items()
        if len(frames) >= 2 and track in depth_classes
    }
    mask = np.isin(live, list(selected))
    packet = {name: arrays[name][mask] for name in DirectFactors.__dataclass_fields__}
    camera_map = {int(frame): i for i, frame in enumerate(keys)}
    packet["view"] = np.array([camera_map[int(frame)] for frame in views[mask]])
    packet["track_id"] = live[mask]
    depth_poses = dict(zip(arrays["camera_ids"], arrays["depth_poses"], strict=True))
    poses = np.array(
        [
            depth_poses[frame]
            @ np.linalg.inv(arrays["T_RcD"][np.flatnonzero(views == frame)[0]])
            for frame in keys
        ]
    )
    seeds = dict(zip(arrays["point_ids"], arrays["point_positions"], strict=True))
    for frame, track, point in zip(
        views[native], live[native], arrays["native_points"][native], strict=True
    ):
        if int(track) not in seeds:
            seeds[int(track)] = transform_points(point, depth_poses[frame])
    packet["points"] = np.array([seeds[int(track)] for track in live[mask]])
    factors = DirectFactors(**packet)
    ids, inverse, points = landmark_layout(factors)
    config = BatchConfig(**report["online_config"]["shared"]["solver"])
    r, JP, JL = compact_linearize(
        poses,
        points,
        factors,
        np.array(report["calibration"]["K_R"]),
        inverse,
        config.depth_model,
    )
    cost, weights = direct_cost(r, factors, config)
    blocks = compact_blocks(
        r, JP, JL, weights, inverse, factors.view, len(ids), len(keys)
    )
    fixed = {camera_map[int(root)] for root in arrays["raw_components"][:, 1]}
    fixed |= set(range(len(keys))) - set(factors.view.tolist())
    if blocks_output is not None:
        if blocks_output.exists():
            raise ValueError("choose a fresh linear block artifact")
        np.savez_compressed(
            blocks_output,
            B=blocks[0],
            gp=blocks[1],
            D=blocks[2],
            gl=blocks[3],
            E=blocks[4],
            views=factors.view,
            inverse=inverse,
            fixed=np.array(sorted(fixed)),
            source_sha256=sha256(path),
        )
    operator = SchurOperator(blocks, factors.view, inverse, (), 0)
    scale = np.sqrt(np.maximum(np.diagonal(operator.diagonal, axis1=1, axis2=2), 1e-15))
    spectrum = np.linalg.eigvalsh(
        operator.diagonal / scale[:, :, None] / scale[:, None, :]
    )
    cameras = [
        {
            "frame": int(frame),
            "retained": int((views == frame).sum()),
            "native": int(((views == frame) & native).sum()),
            "shared_observations": int((factors.view == i).sum()),
            "conditional_normalized_min_eigenvalue": float(spectrum[i, 0]),
            "weak_conditional_block": bool(
                spectrum[i, 0] <= max(1e-10, spectrum[i, -1] / config.max_condition)
            ),
            "max_earlier_shared_native_tracks": max(
                (len(by[frame] & by[earlier]) for earlier in keys if earlier < frame),
                default=0,
            ),
        }
        for i, frame in enumerate(keys)
    ]
    global_information = None
    if global_rank:
        conditioned = SchurOperator(blocks, factors.view, inverse, sorted(fixed), 0)
        dimension = len(conditioned.rhs)
        if dimension > 2048:
            raise ValueError("offline dense rank diagnostic exceeds 2048 dimensions")
        S = np.column_stack([conditioned.multiply(row) for row in np.eye(dimension)])
        S = (S + S.T) / 2
        global_information = normalized_spectrum(S)
        diagonal = np.sqrt(np.maximum(np.diag(S), 1e-15))
        values, vectors = np.linalg.eigh(S / diagonal[:, None] / diagonal[None, :])
        weak = values <= max(1e-10, values[-1] / config.max_condition)
        participation = np.sum(vectors[:, weak] ** 2, axis=1).reshape(-1, 6).sum(axis=1)
        global_information.update(
            weak_directions=int(weak.sum()),
            weak_mode_camera_participation=sorted(
                [
                    {"frame": int(keys[camera]), "weight": float(weight)}
                    for camera, weight in zip(
                        conditioned.free, participation, strict=True
                    )
                ],
                key=lambda value: value["weight"],
                reverse=True,
            )[:20],
            scope="offline fresh undamped camera Schur system with raw component roots and unused cameras fixed, local linearized rank only",
        )
    look = {
        (int(frame), int(track)): i
        for i, (frame, track) in enumerate(zip(views, raw, strict=True))
    }
    contradictions = []
    for source, target, a, b in arrays["associations"]:
        if identities[a] != identities[b]:
            continue
        i, j = look.get((int(source), int(a))), look.get((int(target), int(b)))
        if i is not None and j is not None and np.isfinite(world[[i, j]]).all():
            contradictions.append(
                {
                    "source": int(source),
                    "target": int(target),
                    "native_discrepancy_m": float(np.linalg.norm(world[i] - world[j])),
                }
            )
    region_information = None
    if region is not None:
        inside = (views >= region[0]) & (views < region[1])
        boundary = sorted(set(live[inside]) & set(live[~inside]) & selected)
        positions = np.array([seeds[int(track)] for track in boundary]).reshape(-1, 3)
        region_information = {
            "received_frame_interval": list(region),
            "recorded_cameras": sorted(map(int, set(views[inside]))),
            "boundary_landmarks": [
                {"identity": int(track), "observers": observers[int(track)]}
                for track in boundary
            ],
            "boundary_position_singular_values": np.linalg.svd(
                positions - positions.mean(axis=0), compute_uv=False
            ).tolist()
            if len(positions)
            else [],
            "caveat": "boundary support is diagnostic, its count alone does not certify observability",
        }
    return {
        "source_sha256": sha256(path),
        "source_report_sha256": sha256(run / "report.json"),
        "package_sha256": code_hash(),
        "driver_sha256": sha256(Path(__file__)),
        "cameras": cameras,
        "consecutive_links": links,
        "zero_native_consecutive_links": [
            link for link in links if link["native_tracks"] == 0
        ],
        "raw_component_camera_counts": dict(
            Counter(map(int, arrays["raw_components"][:, 1]))
        ),
        "class_view_support": dict(
            Counter(len(value) for key, value in observers.items() if key in selected)
        ),
        "native_reference_identity_consistency": identity_radii(raw),
        "live_reference_identity_consistency": identity_radii(live),
        "active_loop_reference_discrepancy_m": error_statistics(
            value["native_discrepancy_m"] for value in contradictions
        ),
        "largest_active_loop_discrepancies": sorted(
            contradictions,
            key=lambda value: value["native_discrepancy_m"],
            reverse=True,
        )[:20],
        "exact_saved_objective_cost": cost,
        "global_information": global_information,
        "region_information": region_information,
        "caveat": "conditional blocks do not certify global observability, reference contradictions are not independent physical point truth",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks-output", type=Path)
    parser.add_argument("--global-rank", action="store_true")
    parser.add_argument(
        "--region", type=int, nargs=2, help="received frame interval, stop exclusive"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("choose a fresh diagnostic output")
    args.output.write_text(
        json.dumps(
            diagnose(
                args.run,
                blocks_output=args.blocks_output,
                global_rank=args.global_rank,
                region=args.region,
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
