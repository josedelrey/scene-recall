"""Compare shared color-time projections with historical static tracker checks.

This frozen-state audit does not reconstruct past active tracker decisions.
Reference poses are not used. It measures the chart discrepancy for retained
observations whose live landmarks have a saved shared point estimate.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from scene_recall.geometry.camera import project_points, transform_points
from scene_recall.odometry.evaluation import error_statistics
from scene_recall.run_metadata import code_hash, sha256


def audit(run):
    report = json.loads((run / "report.json").read_text())
    path = run / "shared_estimation.npz"
    arrays = np.load(path)
    live = dict(arrays["live_identities"])
    points = dict(zip(arrays["point_ids"], arrays["point_positions"], strict=True))
    poses = dict(zip(arrays["camera_ids"], arrays["depth_poses"], strict=True))
    atlas = np.load(run / "atlas.npz")
    selected = np.array(
        [
            i
            for i, track in enumerate(arrays["track_id"])
            if live[track] in points
            and atlas["component_id"][arrays["view"][i]] == atlas["component_id"][0]
        ],
        dtype=int,
    )
    views = arrays["view"][selected]
    world = np.array([points[live[t]] for t in arrays["track_id"][selected]]).reshape(
        -1, 3
    )
    local = np.empty_like(world)
    for frame in np.unique(views):
        mask = views == frame
        local[mask] = transform_points(world[mask], np.linalg.inv(poses[frame]))
    exposure = arrays["T_RcD"][selected]
    actual = np.einsum("nij,nj->ni", exposure[:, :3, :3], local) + exposure[:, :3, 3]
    calibration = report["calibration"]
    virtual = transform_points(local, np.array(calibration["T_RD"]))
    K = np.array(calibration["K_R"])
    a, b = project_points(actual, K), project_points(virtual, K)
    shift = np.linalg.norm(a - b, axis=1)
    observed = arrays["pixels"][selected]
    ra, rb = np.linalg.norm(a - observed, axis=1), np.linalg.norm(b - observed, axis=1)
    valid = np.isfinite(shift) & np.isfinite(ra) & np.isfinite(rb)
    return {
        "package_sha256": code_hash(),
        "driver_sha256": sha256(Path(__file__)),
        "source_sha256": sha256(path),
        "source_report_sha256": sha256(run / "report.json"),
        "root_observations": len(selected),
        "finite_projections": int(valid.sum()),
        "projection_shift_px": error_statistics(shift[valid]),
        "shared_color_reprojection_px": error_statistics(ra[valid]),
        "static_tracker_reprojection_px": error_statistics(rb[valid]),
        "shared_within_3px_static_outside": int(((ra <= 3) & (rb > 3) & valid).sum()),
        "scope": "frozen final root-connected shared landmark observations, static tracker versus retained exposure projection, no reference poses, not an audit of actual historical accept/reject decisions",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("choose a fresh exposure audit output")
    args.output.write_text(
        json.dumps(audit(args.run), indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
