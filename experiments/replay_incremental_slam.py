"""Post hoc full relinearization of exactly the saved incremental raw factors.

This diagnostic keeps identities, RGB/depth bindings, exposure motion and weights
fixed. It tests whether local marginalization alone explains an observed error.
The saved estimated trajectory and points initialize the fit. References score
the result only. This is not an online candidate or a new association method.
"""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from scene_recall.multiview.batch_ba import BatchConfig, solve_batch
from scene_recall.multiview.fixed_landmarks import DirectFactors
from scene_recall.odometry.evaluation import evaluate_trajectory
from scene_recall.odometry.io import load_trajectory, save_run
from scene_recall.odometry.pipeline import PairEstimate
from scene_recall.run_metadata import code_hash, sha256
from scene_recall.slam.system import RevisionArchive


def replay(run, output):
    if output.exists():
        raise ValueError("choose a fresh output directory")
    report, rows = load_trajectory(run)
    if report["evaluation"]["segment_count"] != 1:
        raise ValueError("diagnostic requires one continuous segment")
    with np.load(run / "measurements.npz", allow_pickle=False) as a:
        points = dict(zip(a["final_track_ids"], a["final_points"], strict=True))
        arrays = {key: a[key].copy() for key in DirectFactors.__dataclass_fields__}
    camera_ids = sorted(set(map(int, arrays["view"])))
    arrays["view"] = np.array([camera_ids.index(int(key)) for key in arrays["view"]])
    arrays["points"] = np.array([points[int(key)] for key in arrays["track_id"]])
    factors = DirectFactors(**arrays)
    transforms = {
        key: factors.T_RcD[np.flatnonzero(factors.view == i)[0]]
        for i, key in enumerate(camera_ids)
    }
    poses = np.array(
        [rows[key].T_SC_D @ np.linalg.inv(transforms[key]) for key in camera_ids]
    )
    config = BatchConfig(depth_model=report["config"]["solver"]["depth_model"])
    fitted, points, diagnostic = solve_batch(
        poses, factors, np.array(report["calibration"]["K_R"]), config
    )
    archive = RevisionArchive()
    for row in rows:
        archive.append(
            row.T_SC_D,
            keyframe=row.frame_index in transforms,
            boundary=row.status != "tracked",
        )
    dirty = archive.apply(
        {
            key: pose @ transforms[key]
            for key, pose in zip(camera_ids, fitted, strict=True)
        }
    )
    changed = []
    for i, row in enumerate(rows):
        pair = row.pair
        if row.status == "tracked":
            pair = PairEstimate(np.linalg.inv(archive.poses[i]) @ archive.poses[i - 1])
        changed.append(replace(row, T_SC_D=archive.poses[i], pair=pair))
    evaluation = evaluate_trajectory(changed, (1, 5, 10, 20, 50, 100))
    provenance = {
        "report_sha256": sha256(run / "report.json"),
        "trajectory_sha256": sha256(run / "trajectory.npz"),
        "measurements_sha256": sha256(run / "measurements.npz"),
        "package_sha256": code_hash(),
        "driver_sha256": sha256(Path(__file__)),
    }
    new_report = {
        **report,
        "evaluation": evaluation,
        "experiment": {
            "type": "post hoc complete relinearization of fixed incremental factors",
            "inputs": provenance,
            "batch_config": asdict(config),
            "solver": diagnostic,
            "dirty_frames": dirty,
        },
    }
    save_run(output, changed, new_report, report["source"]["start"])
    np.savez_compressed(
        output / "landmarks.npz", track_ids=np.unique(factors.track_id), points=points
    )
    result = {
        "inputs": provenance,
        "solver": diagnostic,
        "before": report["evaluation"],
        "after": evaluation,
    }
    (output / "replay_comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(diagnostic["status"], diagnostic.get("reason"), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    replay(args.run, args.output_dir)


if __name__ == "__main__":
    main()
