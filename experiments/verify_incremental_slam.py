"""Verify baseline preservation, raw/cached equality and revision-aware prefixes."""

import argparse
import ast
import json
from pathlib import Path

import numpy as np

from scene_recall.run_metadata import sha256


def equal(a, b):
    return (
        bool(np.array_equal(a, b))
        if a.dtype.kind in "US"
        else bool(np.array_equal(a, b, equal_nan=True))
    )


def compare_archives(first, second, *, length=None):
    with (
        np.load(first, allow_pickle=False) as a,
        np.load(second, allow_pickle=False) as b,
    ):
        if set(a.files) != set(b.files):
            raise ValueError("archive schemas differ")
        return {
            key: equal(a[key] if length is None else a[key][:length], b[key])
            for key in a.files
        }


def without_timing(value):
    if isinstance(value, dict):
        return {
            key: without_timing(v)
            for key, v in value.items()
            if key != "seconds" and not key.endswith("_seconds")
        }
    if isinstance(value, list):
        return [without_timing(v) for v in value]
    return value


def verify(root):
    protected = json.loads((root / "protected_baselines.json").read_text())
    results = {
        "protected_files": len(protected),
        "protected_unchanged": all(
            sha256(Path(path)) == digest for path, digest in protected.items()
        ),
        "repeats": {},
        "sparse": {},
        "full_prefixes": {},
    }
    for policy in ("control", "native", "axial", "static"):
        first, second = (
            root / "final_repeat_cached" / policy,
            root / "final_repeat_raw" / policy,
        )
        arrays = {
            name: compare_archives(first / name, second / name)
            for name in ("trajectory.npz", "emitted_trajectory.npz", "measurements.npz")
        }
        a, b = (json.loads((p / "report.json").read_text()) for p in (first, second))
        result = {
            "arrays": arrays,
            "diagnostics_exact": [without_timing(r["diagnostics"]) for r in a["frames"]]
            == [without_timing(r["diagnostics"]) for r in b["frames"]],
            "evaluations_exact": a["evaluation"] == b["evaluation"]
            and a["emitted_evaluation"] == b["emitted_evaluation"],
        }
        results["repeats"][policy] = result
        full = root / "full_sequences/xyz" / policy
        with (
            np.load(full / "emitted_trajectory.npz") as x,
            np.load(first / "emitted_trajectory.npz") as y,
        ):
            emitted_exact = equal(x["T_SC_D"][:80], y["T_SC_D"])
        with (
            np.load(full / "trajectory.npz") as x,
            np.load(first / "trajectory.npz") as y,
        ):
            sealed_exact = equal(x["T_SC_D"][:40], y["T_SC_D"][:40])
        results["full_prefixes"][policy] = {
            "emitted_first80_exact": emitted_exact,
            "sealed_first40_exact": sealed_exact,
        }
    for name in ("desk", "rpy"):
        original = (
            root.parent / "local_map" / (name + "_full") / "budget/trajectory.npz"
        )
        results["sparse"][name] = compare_archives(
            original, root / "full_cached_inputs" / name / "budget/trajectory.npz"
        )
    results["sparse"]["xyz80"] = compare_archives(
        root.parent / "local_map/xyz_full/budget/trajectory.npz",
        root / "timing_inputs/budget/trajectory.npz",
        length=80,
    )
    audit = {}
    for name in ("matrix", "independent", "full_sequences"):
        for path in (root / name).glob("*/*/report.json"):
            report = json.loads(path.read_text())
            config = report.get("config", {})
            if "window_size" not in config:
                continue
            last, expired = None, []
            for row in report["frames"]:
                diagnostic, frame = row["diagnostics"], row["frame_index"]
                if row["status"] == "lost":
                    last = None
                if (
                    "exposure_status" in diagnostic
                    and last is not None
                    and frame - last
                    >= config["window_size"] * config["keyframe_interval"]
                ):
                    expired.append(frame)
                if "graph_restart" in diagnostic:
                    last = None
                if "measurement_count" in diagnostic:
                    last = frame
            audit[str(path)] = expired
    results["unaffected_matrix_and_independent"] = all(
        not frames
        for path, frames in audit.items()
        if "/matrix/" in path or "/independent/" in path
    )
    results["prototype_runs_requiring_lifecycle_replacement"] = [
        path for path, frames in audit.items() if frames
    ]

    class WithoutExpiry(ast.NodeTransformer):
        def visit_If(self, node):
            self.generic_visit(node)
            if any(
                isinstance(statement, ast.Assign)
                and isinstance(statement.value, ast.Constant)
                and statement.value.value == "no measurements within lag horizon"
                for statement in node.body
            ):
                return None
            return node

    package = Path(__file__).parents[1] / "src/scene_recall/slam"
    old = ast.parse((root / "prototype_sources/system.py").read_text())
    current = WithoutExpiry().visit(ast.parse((package / "system.py").read_text()))
    results["system_change_only_lag_expiry"] = ast.dump(old) == ast.dump(current)
    results["joint_solver_unchanged"] = sha256(
        root / "prototype_sources/graph.py"
    ) == sha256(package / "graph.py")
    checks = [
        results["system_change_only_lag_expiry"],
        results["joint_solver_unchanged"],
        results["protected_unchanged"],
        results["unaffected_matrix_and_independent"],
    ]
    checks += [
        value for result in results["sparse"].values() for value in result.values()
    ]
    checks += [
        value
        for result in results["full_prefixes"].values()
        for value in result.values()
    ]
    for result in results["repeats"].values():
        checks += [result["diagnostics_exact"], result["evaluations_exact"]]
        checks += [
            value for archive in result["arrays"].values() for value in archive.values()
        ]
    results["all_passed"] = all(checks)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a fresh output file")
    report = verify(args.root)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("all checks passed", report["all_passed"], flush=True)
    if not report["all_passed"]:
        raise ValueError("incremental verification failed")


if __name__ == "__main__":
    main()
