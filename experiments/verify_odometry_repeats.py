"""Verify repeated prefixes or clips against saved full odometry runs.

Pair estimates must match exactly. Local trajectories in later clips must match
after an estimated-pose coordinate change, without any reference alignment.
Optionally fingerprint source tables and every PNG for future reproduction.
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np


def verify_clip(full, repeat):
    original = json.loads((full / "report.json").read_text())
    rerun = json.loads((repeat / "report.json").read_text())
    for key in ("backend", "config", "calibration"):
        if original[key] != rerun[key]:
            raise ValueError(f"repeat differs: {key}")
    for key in (
        "code_sha256",
        "python",
        "numpy",
        "opencv",
        "opencv_threads",
        "openblas_num_threads_env",
        "omp_num_threads_env",
    ):
        if original["environment"][key] != rerun["environment"][key]:
            raise ValueError(f"repeat environment differs: {key}")
    for key in ("path", "sequence_id", "association_config", "adapter"):
        if original["source"][key] != rerun["source"][key]:
            raise ValueError(f"repeat source differs: {key}")
    start = rerun["source"]["start"] - original["source"]["start"]
    count = rerun["evaluation"]["frame_count"]
    exact = []
    with (
        np.load(full / "trajectory.npz") as first,
        np.load(repeat / "trajectory.npz") as second,
    ):
        for key in second.files:
            expected = first[key][start : start + count]
            actual = second[key]
            if start == 0:
                same = (
                    np.array_equal(expected, actual, equal_nan=True)
                    if actual.dtype.kind in "fc"
                    else np.array_equal(expected, actual)
                )
                if not same:
                    raise ValueError(f"prefix differs: {key}")
                exact.append(key)
            elif key in (
                "T_C1C0",
                "initial_T_C1C0",
                "candidate_T_C1C0",
                "ransac_T_C1C0",
                "status",
            ):
                same = (
                    np.array_equal(expected[1:], actual[1:], equal_nan=True)
                    if actual.dtype.kind in "fc"
                    else np.array_equal(expected[1:], actual[1:])
                )
                if not same:
                    raise ValueError(f"clip pair differs: {key}")
                exact.append(key + "[1:]")
            elif key in (
                "timestamp_s",
                "source_frame_index",
                "T_WC_D_reference",
                "reference_present",
            ):
                same = (
                    np.array_equal(expected, actual, equal_nan=True)
                    if actual.dtype.kind in "fc"
                    else np.array_equal(expected, actual)
                )
                if not same:
                    raise ValueError(f"clip input differs: {key}")
                exact.append(key)
        if not np.array_equal(
            second["segment_id"],
            first["segment_id"][start : start + count] - first["segment_id"][start],
        ):
            raise ValueError("clip segment boundaries differ")
        discrepancy = 0.0
        for segment in np.unique(second["segment_id"]):
            indices = np.flatnonzero(second["segment_id"] == segment)
            anchor = indices[0]
            expected = (
                np.linalg.inv(first["T_SC_D"][start + anchor])
                @ first["T_SC_D"][start + indices]
            )
            discrepancy = max(
                discrepancy, float(np.max(abs(expected - second["T_SC_D"][indices])))
            )
        if discrepancy > 1e-12:
            raise ValueError("rebased clip trajectory differs")
    for index, frame in enumerate(rerun["frames"]):
        previous = original["frames"][start + index]
        if frame["provenance"] != previous["provenance"]:
            raise ValueError("repeat provenance differs")
        if index or start == 0:
            clean = lambda diagnostics: {
                k: v for k, v in diagnostics.items() if not k.endswith("_seconds")
            }
            if (
                clean(frame["diagnostics"]) != clean(previous["diagnostics"])
                or frame["reason"] != previous["reason"]
            ):
                raise ValueError("repeat numerical diagnostics differ")
    return {
        "full": str(full.resolve()),
        "repeat": str(repeat.resolve()),
        "start": rerun["source"]["start"],
        "frames": count,
        "exact_arrays": exact,
        "numerical_diagnostics_equal": True,
        "segment_boundaries_equal": True,
        "rebased_trajectory_max_abs_difference": discrepancy,
        "full_report_sha256": hashlib.sha256(
            (full / "report.json").read_bytes()
        ).hexdigest(),
        "repeat_report_sha256": hashlib.sha256(
            (repeat / "report.json").read_bytes()
        ).hexdigest(),
    }


def source_manifest(source):
    """Fingerprint original tables and all images named in the RGB/depth tables."""
    paths = {"rgb.txt", "depth.txt", "groundtruth.txt"}
    for table in ("rgb.txt", "depth.txt"):
        for line in (source / table).read_text().splitlines():
            fields = line.split("#", 1)[0].split()
            if fields:
                paths.add(fields[1])
    files = []
    digest = hashlib.sha256()
    for relative in sorted(paths):
        path = (source / relative).resolve()
        if not path.is_relative_to(source.resolve()):
            raise ValueError("manifest path leaves source directory")
        data = path.read_bytes()
        row = {
            "path": relative,
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        files.append(row)
        digest.update(json.dumps(row, sort_keys=True).encode())
    return {
        "source": str(source.resolve()),
        "files": files,
        "manifest_sha256": digest.hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-dir", type=Path, required=True)
    parser.add_argument("--repeat-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--hash-sources", action="store_true")
    args = parser.parse_args()
    full = json.loads((args.full_dir / "comparison.json").read_text())["ranges"]
    runs = {
        (row["source"]["sequence_id"], name): Path(run["directory"])
        for row in full
        for name, run in row["runs"].items()
    }
    checks = []
    for directory in args.repeat_dir:
        for row in json.loads((directory / "comparison.json").read_text())["ranges"]:
            for name, run in row["runs"].items():
                checks.append(
                    verify_clip(
                        runs[row["source"]["sequence_id"], name], Path(run["directory"])
                    )
                )
    result = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "verification_host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_model": next(
                (
                    line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")
                ),
                None,
            )
            if Path("/proc/cpuinfo").exists()
            else None,
        },
        "checks": checks,
    }
    (args.full_dir / "reproducibility.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    if args.hash_sources:
        manifests = [source_manifest(Path(row["source"]["path"])) for row in full]
        (args.full_dir / "source_manifest.json").write_text(
            json.dumps(manifests, indent=2) + "\n"
        )
    print(f"Verified {len(checks)} repeated runs")


if __name__ == "__main__":
    main()
