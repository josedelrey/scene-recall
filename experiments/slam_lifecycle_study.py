"""Controls, serial timing and provenance audits for landmark retirement research.

The accuracy matrix uses run_incremental_slam.py. This utility supplies matched
no-optimizer controls and verifies retained evidence without rewriting old runs.
"""

import argparse
import json
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from incremental_slam import evaluate_maps, run_case, variants
from summarize_incremental_slam import maps, motion

from scene_recall.odometry.evaluation import error_statistics
from scene_recall.odometry.io import load_trajectory
from scene_recall.run_metadata import code_hash, sha256


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def equality(left, right):
    with (
        np.load(left, allow_pickle=False) as a,
        np.load(right, allow_pickle=False) as b,
    ):
        return set(a.files) == set(b.files) and all(
            np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind in "fc")
            for k in a.files
        )


def controls(baseline, output):
    policies = variants()
    policies["turnover_control"] = replace(policies["turnover"], optimize=False)
    with patch("incremental_slam.variants", return_value=policies):
        run_case(baseline, output, ["control", "turnover_control"])
    result = {
        name: equality(output / "control" / name, output / "turnover_control" / name)
        for name in ("trajectory.npz", "emitted_trajectory.npz")
    }
    write(output / "equivalence.json", result)
    if not all(result.values()):
        raise ValueError("lifecycle changed poses without optimizer feedback")


def timing(baseline, output):
    if output.exists():
        raise ValueError("choose a fresh output directory")
    output.mkdir(parents=True)
    forward = ("native", "sqrt", "turnover", "persistent", "persistent_turnover")
    orders = [forward, tuple(reversed(forward))]
    write(
        output / "protocol.json",
        {
            "orders": orders,
            "implementation": code_hash(),
            "driver_sha256": sha256(Path(__file__)),
            "baseline_report_sha256": sha256(baseline / "report.json"),
            "scope": "Serial raw input, warm files, one policy per phase. Run after other jobs.",
            "cpu": next(
                (
                    line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")
                ),
                "unknown",
            )
            if Path("/proc/cpuinfo").exists()
            else "unknown",
        },
    )
    result = {}
    for direction, order in zip(("forward", "reverse"), orders, strict=True):
        for name in order:
            path = output / (direction + "_" + name)
            if name.startswith("persistent"):
                from slam_persistent import run as persistent_run

                run = persistent_run(baseline, path, [name])
            else:
                run = run_case(baseline, path, [name])
            report, _ = load_trajectory(path / name)
            ds = [r["diagnostics"] for r in report["frames"]]
            result[path.name] = {
                "fps": len(ds) / run["wall_seconds"],
                "keyframe_seconds": error_statistics(
                    d["total_seconds"] for d in ds if "measurement_seconds" in d
                ),
                "measurement_seconds": error_statistics(
                    d["measurement_seconds"] for d in ds if "measurement_seconds" in d
                ),
                "solver_seconds": error_statistics(
                    d["solver"]["seconds"]
                    for d in ds
                    if "seconds" in d.get("solver", {})
                ),
            }
            print(path.name, result[path.name]["fps"], flush=True)
    write(output / "report.json", result)


def summarize(manifest, output):
    cases = json.loads(manifest.read_text())
    result = {"manifest_sha256": sha256(manifest), "cases": {}}
    for case, paths in cases.items():
        entries = {}
        source_identity = None
        for name, path in paths.items():
            path = Path(path)
            report, rows = load_trajectory(path)
            identity = (report["source"], report["calibration"])
            if source_identity is not None and identity != source_identity:
                raise ValueError("candidate and control sources or calibration differ")
            source_identity = identity
            ds = [r["diagnostics"] for r in report["frames"]]
            entries[name] = {
                "path": str(path),
                "report_sha256": sha256(path / "report.json"),
                "trajectory_sha256": sha256(path / "trajectory.npz"),
                "final": motion(report),
                "loss_indices": [r.frame_index for r in rows if r.status == "lost"],
            }
            if report["backend"] == "incremental_slam":
                marginals = [
                    d[k]
                    for d in ds
                    for k in ("marginalization", "point_marginalization")
                    if d.get(k)
                ]
                entries[name].update(
                    emitted=motion(report, emitted=True),
                    consistency=report["measurement_consistency"],
                    optimizer_statuses=dict(
                        Counter(d["solver"]["status"] for d in ds if "solver" in d)
                    ),
                    optimizer_reasons=dict(
                        Counter(d["solver"]["reason"] for d in ds if "solver" in d)
                    ),
                    graph_restarts=dict(
                        Counter(d["graph_restart"] for d in ds if "graph_restart" in d)
                    ),
                    admissions=sum("measurement_count" in d for d in ds),
                    raw_observations_consumed=sum(
                        m["consumed_observations"] for m in marginals
                    ),
                    rank_deficient_retirements=sum(
                        m.get("eliminated_rank", 0) < m.get("eliminated_dimension", 0)
                        for m in marginals
                    ),
                    max_active_cameras=max(d.get("active_cameras", 0) for d in ds),
                    max_active_points=max(d.get("active_points", 0) for d in ds),
                )
        # Separate segment populations cannot be ranked by their largest-segment ATE.
        cohorts = [tuple(v["loss_indices"]) for v in entries.values()]
        result["cases"][case] = {
            "same_loss_boundaries": len(set(cohorts)) == 1,
            "variants": entries,
        }
        # Present existing pairwise fusion runs under one read-only view so the
        # common-mask scorer verifies identical targets, observations and controls.
        with tempfile.TemporaryDirectory(prefix="slam-lifecycle-summary-") as directory:
            shared = Path(directory)
            names = [n for n in paths if n != "budget"]
            for name in names:
                path = Path(paths[name]).resolve()
                (shared / name).symlink_to(path, target_is_directory=True)
                for kind in ("maps", "tsdf"):
                    source = path.parent / (kind + "_" + name)
                    if source.exists():
                        (shared / (kind + "_" + name)).symlink_to(
                            source, target_is_directory=True
                        )
            for kind in ("surface", "tsdf"):
                try:
                    result["cases"][case][kind] = maps(shared, names, kind)
                except ValueError as error:
                    result["cases"][case][kind] = {
                        "status": "withheld",
                        "reason": str(error),
                    }
                scored = result["cases"][case][kind]
                scored["all_policy_complete"] = set(names).issubset(
                    scored.get("scores", {})
                )
    write(output, result)


def verify(root):
    protected = json.loads((root / "protected.json").read_text())
    changed = [p for p, expected in protected.items() if sha256(Path(p)) != expected]
    records = []
    for p in root.glob("**/comparison.json"):
        if "development" in p.parts:
            continue
        comparison = json.loads(p.read_text())
        for name, entry in comparison.get("variants", {}).items():
            path = p.parent / name
            if (
                sha256(path / "report.json") != entry["report_sha256"]
                or sha256(path / "trajectory.npz") != entry["trajectory_sha256"]
            ):
                raise ValueError("candidate evidence changed")
            report, rows = load_trajectory(path)
            with np.load(path / "measurements.npz", allow_pickle=False) as a:
                if "view" in a:
                    valid = a["native_index"] >= 0
                    for column, mask in (
                        ("feature_index", np.ones(len(valid), bool)),
                        ("native_index", valid),
                    ):
                        pairs = np.column_stack((a["view"][mask], a[column][mask]))
                        if len(np.unique(pairs, axis=0)) != len(pairs):
                            raise ValueError("raw observation reused")
            with np.load(path / "trajectory.npz", allow_pickle=False) as a:
                for i, row in enumerate(rows):
                    if row.status == "tracked":
                        np.testing.assert_allclose(
                            a["T_C1C0"][i],
                            np.linalg.inv(row.T_SC_D) @ rows[i - 1].T_SC_D,
                            atol=1e-12,
                        )
            records.append(
                {
                    "path": str(path),
                    "frames": len(rows),
                    "implementation": report["environment"]["code_sha256"],
                }
            )
    result = {
        "protected_files": len(protected),
        "changed_protected_files": changed,
        "validated_runs": records,
        "current_implementation": code_hash(),
    }
    write(root / "verification.json", result)
    if changed:
        raise ValueError("protected evidence changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("controls", "timing"):
        p = sub.add_parser(name)
        p.add_argument("baseline", type=Path)
        p.add_argument("output", type=Path)
    p = sub.add_parser("summary")
    p.add_argument("manifest", type=Path)
    p.add_argument("output", type=Path)
    p = sub.add_parser("verify")
    p.add_argument("root", type=Path)
    p = sub.add_parser("maps")
    p.add_argument("baseline", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--variants", nargs="+", required=True)
    p = sub.add_parser("outages")
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--persistent", action="store_true")
    args = parser.parse_args()
    if args.action in ("controls", "timing"):
        globals()[args.action](args.baseline, args.output)
    elif args.action == "summary":
        summarize(args.manifest, args.output)
    elif args.action == "maps":
        evaluate_maps(args.baseline, args.output, args.variants)
    elif args.action == "outages":
        from slam_outages import run

        if args.persistent:
            from slam_persistent import PersistentSeedsSLAM

            run(
                args.source,
                args.output,
                configurations={
                    "persistent": variants()["native"],
                    "persistent_turnover": variants()["turnover"],
                },
                engine_factory=PersistentSeedsSLAM,
            )
        else:
            run(
                args.source,
                args.output,
                configurations={
                    n: variants()[n] for n in ("control", "native", "sqrt", "turnover")
                },
            )
    else:
        verify(args.root)


if __name__ == "__main__":
    main()
