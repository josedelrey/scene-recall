"""Serial, matched-input comparisons of sparse, depth-only, and hybrid odometry.

Run each backend with default settings and one OpenCV/BLAS thread. Backend
order rotates across ranges. Reuse reports with --summarize-only after a run.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from scene_recall.odometry.evaluation import error_statistics, pose_error
from scene_recall.odometry.pipeline import is_rigid

BACKENDS = ("sparse", "icp", "hybrid")
DEFAULT_RANGES = ("0:200", "1000:1200", "2000:2200", "3500:3700", "4500:4700")
ROOT = Path(__file__).resolve().parents[1]


def summarize_range(run_dirs):
    """Compare accuracy on exactly the same reference-valid, continuous pairs."""
    reports = {
        name: json.loads((run / "report.json").read_text())
        for name, run in run_dirs.items()
    }
    archives = {
        name: np.load(run / "trajectory.npz", allow_pickle=False)
        for name, run in run_dirs.items()
    }
    try:
        first = reports["sparse"]
        for name in BACKENDS:
            if reports[name]["backend"] != name:
                raise ValueError("report backend differs from comparison label")
            if (
                reports[name]["source"] != first["source"]
                or reports[name]["calibration"] != first["calibration"]
            ):
                raise ValueError(
                    "comparisons require identical source ranges and calibration"
                )
            for key in (
                "python",
                "numpy",
                "opencv",
                "opencv_threads",
                "openblas_num_threads_env",
                "omp_num_threads_env",
                "code_sha256",
            ):
                if reports[name]["environment"][key] != first["environment"][key]:
                    raise ValueError(f"comparison environments differ: {key}")
            for key in (
                "frame_index",
                "source_frame_index",
                "T_WC_D_reference",
                "reference_present",
            ):
                if not np.array_equal(
                    archives[name][key], archives["sparse"][key], equal_nan=True
                ):
                    raise ValueError(f"comparison archives differ: {key}")
        if (
            reports["hybrid"]["config"]["sparse"] != reports["sparse"]["config"]
            or reports["hybrid"]["config"]["icp"] != reports["icp"]["config"]
        ):
            raise ValueError("hybrid component settings must match standalone runs")
        count = len(archives["sparse"]["frame_index"])
        references = archives["sparse"]["T_WC_D_reference"]
        common_rpe = {}
        for interval in first["evaluation"]["rpe_by_frame_interval"]:
            errors = {name: [] for name in BACKENDS}
            k = int(interval)
            for source in range(max(0, count - k)):
                target = source + k
                if not all(
                    archive["segment_id"][source] == archive["segment_id"][target]
                    for archive in archives.values()
                ):
                    continue
                if not is_rigid(references[source]) or not is_rigid(references[target]):
                    continue
                reference = np.linalg.inv(references[target]) @ references[source]
                if not is_rigid(reference):
                    continue
                for name, archive in archives.items():
                    estimate = (
                        np.linalg.inv(archive["T_SC_D"][target])
                        @ archive["T_SC_D"][source]
                    )
                    errors[name].append(pose_error(estimate, reference))
            common_rpe[interval] = {
                name: {
                    "translation_m": error_statistics(e[0] for e in rows),
                    "rotation_deg": error_statistics(e[1] for e in rows),
                }
                for name, rows in errors.items()
            }
        outcomes = []
        hybrid = reports["hybrid"]
        for frame, errors in zip(
            hybrid["frames"][1:], hybrid["evaluation"]["pair_errors_m_deg"]
        ):
            diagnostics = frame["diagnostics"]
            if (
                diagnostics.get("geometric_refinement") != "refined"
                or errors["initial"] is None
                or errors["final"] is None
            ):
                continue
            before = diagnostics["icp_common_test_initial_rmse_m"]
            after = diagnostics["icp_common_test_final_rmse_m"]
            outcomes.append(
                {
                    "source_target_index": frame["source_frame_index"],
                    "initial_error_m_deg": errors["initial"],
                    "final_error_m_deg": errors["final"],
                    "common_test_support": diagnostics["icp_common_test_support"],
                    "depth_rmse_initial_m": before,
                    "depth_rmse_final_m": after,
                    "depth_residual_reduced": None
                    if before is None or after is None
                    else after < before,
                    "translation_improved": errors["final"][0] < errors["initial"][0],
                    "rotation_improved": errors["final"][1] < errors["initial"][1],
                }
            )
        result = {
            "source": first["source"],
            "common_rpe": common_rpe,
            "runs": {
                name: {
                    "directory": str(run_dirs[name]),
                    "report_sha256": hashlib.sha256(
                        (run_dirs[name] / "report.json").read_bytes()
                    ).hexdigest(),
                    "config": report["config"],
                    "timing": report["timing"],
                    "evaluation": report["evaluation"],
                    "failures_by_reason": dict(
                        Counter(
                            frame["reason"]
                            for frame in report["frames"]
                            if frame["status"] == "lost"
                        )
                    ),
                    "geometric_refinement_statuses": dict(
                        Counter(
                            frame["diagnostics"].get(
                                "geometric_refinement", "not_applicable"
                            )
                            for frame in report["frames"][1:]
                        )
                    ),
                    "icp_finest_statuses": dict(
                        Counter(
                            frame["diagnostics"].get("icp_level_1_status", "not_run")
                            for frame in report["frames"][1:]
                        )
                    ),
                    "refinement_failures_by_reason": dict(
                        Counter(
                            frame["diagnostics"]["geometric_refinement_reason"]
                            for frame in report["frames"]
                            if frame["diagnostics"].get("geometric_refinement_reason")
                            is not None
                        )
                    ),
                }
                for name, report in reports.items()
            },
            "hybrid_refinement_outcomes": outcomes,
            "residual_reduced_but_translation_worsened": sum(
                row["depth_residual_reduced"] is True
                and not row["translation_improved"]
                for row in outcomes
            ),
            "residual_reduced_but_rotation_worsened": sum(
                row["depth_residual_reduced"] is True and not row["rotation_improved"]
                for row in outcomes
            ),
        }
        # Verify that the hybrid initialization is the complete standalone sparse result.
        valid = np.isfinite(archives["hybrid"]["initial_T_C1C0"]).all(axis=(1, 2))
        result["hybrid_sparse_initialization_max_abs_difference"] = (
            float(
                np.max(
                    abs(
                        archives["hybrid"]["initial_T_C1C0"][valid]
                        - archives["sparse"]["T_C1C0"][valid]
                    )
                )
            )
            if valid.any()
            else None
        )
        return result
    finally:
        for archive in archives.values():
            archive.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sens-path", type=Path, nargs="+", required=True)
    parser.add_argument("--ranges", nargs="+", default=DEFAULT_RANGES)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    ranges = []
    for value in args.ranges:
        try:
            start, stop = map(int, value.split(":"))
            if start < 0 or stop <= start:
                raise ValueError
        except ValueError:
            parser.error("ranges must use start:stop with 0 <= start < stop")
        ranges.append((start, stop))
    if len(set(ranges)) != len(ranges):
        parser.error("ranges must be unique")
    sources = [source.expanduser().resolve() for source in args.sens_path]
    if len({source.stem for source in sources}) != len(sources):
        parser.error("captures must have unique names")
    output = args.output_dir.expanduser().resolve()
    if not args.summarize_only:
        if output.exists():
            parser.error("output directory exists, choose a fresh directory")
        output.mkdir(parents=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    results = []
    for source in sources:
        for order_index, (start, stop) in enumerate(ranges):
            run_dirs = {
                name: output / f"{source.stem}_{start}_{stop}_{name}"
                for name in BACKENDS
            }
            if not args.summarize_only:
                order = BACKENDS[order_index % 3 :] + BACKENDS[: order_index % 3]
                for name in order:
                    subprocess.run(
                        [
                            sys.executable,
                            str(ROOT / "scripts" / "scannet_odometry.py"),
                            "--sens-path",
                            str(source),
                            "--start",
                            str(start),
                            "--stop",
                            str(stop),
                            "--backend",
                            name,
                            "--output-dir",
                            str(run_dirs[name]),
                        ],
                        cwd=ROOT,
                        env=env,
                        check=True,
                    )
            results.append(summarize_range(run_dirs))
            print(f"Compared {source.stem} [{start}, {stop})", flush=True)
    report = {
        "invocation": sys.argv,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "method": "serial default backends, rotating order, identical inputs, common continuous RPE populations",
        "ranges": results,
    }
    (output / "comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
