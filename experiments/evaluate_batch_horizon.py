"""Separate reconstruction-horizon diagnostic without changing optimizer inputs."""

import argparse
import copy
import json
import shutil
from pathlib import Path

from scene_recall.mapping.heldout import heldout_split
from scene_recall.mapping.voxel import MappingConfig
from scene_recall.multiview.pose_evaluation import evaluate_pose_maps
from scene_recall.odometry.io import load_trajectory
from scene_recall.run_metadata import sha256


def evaluate_horizon(
    run, output, budget, joint_run=None, allow_partial_reference=False
):
    if output.exists() or output.is_symlink():
        raise ValueError("horizon output already exists")
    original = json.loads((run / "report.json").read_text(encoding="utf-8"))
    report = copy.deepcopy(original)
    report["mapping_config"]["max_keyframes_per_submap"] = budget
    _, rows = load_trajectory(Path(report["baseline_run"]))
    report["split"] = heldout_split(
        rows, MappingConfig(**report["mapping_config"]), report["offset"]
    )
    if report["split"]["train_indices"] != original["split"]["train_indices"]:
        raise AssertionError("horizon diagnostic changed training observations")
    report["horizon_parent"] = str(run.resolve())
    report["horizon_parent_report_sha256"] = sha256(run / "report.json")
    report["horizon_policy"] = (
        "post hoc reconstruction extent diagnostic, unchanged optimizer and training views"
    )
    report["variants"] = {
        name: report["variants"][name] for name in ("clock", "timed", "batch")
    }
    sources = {name: run for name in report["variants"]}
    if joint_run is not None:
        joint = json.loads((joint_run / "report.json").read_text(encoding="utf-8"))
        for key in (
            "baseline_report_sha256",
            "baseline_trajectory_sha256",
            "training_measurements_sha256",
            "native_measurements_sha256",
            "offset",
        ):
            if joint[key] != report[key]:
                raise ValueError("joint control has different inputs")
        report["variants"]["joint"] = joint["variants"]["joint"]
        sources["joint"] = joint_run
        report["joint_parent"] = str(joint_run.resolve())
        report["joint_parent_report_sha256"] = sha256(joint_run / "report.json")
    output.mkdir(parents=True)
    for file in ("training_measurements.npz", "native_measurements.npz"):
        shutil.copyfile(run / file, output / file)
    for name, source in sources.items():
        shutil.copytree(source / name, output / name)
        if "factor_archive_sha256" in report["variants"][name]:
            shutil.copyfile(
                source / f"{name}_factors.npz", output / f"{name}_factors.npz"
            )
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return evaluate_pose_maps(
        output,
        output / "maps",
        variants=tuple(sources),
        allow_partial_training_reference=allow_partial_reference,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--map-budget", type=int, required=True)
    parser.add_argument("--joint-run", type=Path)
    parser.add_argument("--allow-partial-training-reference", action="store_true")
    args = parser.parse_args()
    evaluate_horizon(
        args.batch_run,
        args.output_dir,
        args.map_budget,
        args.joint_run,
        args.allow_partial_training_reference,
    )


if __name__ == "__main__":
    main()
