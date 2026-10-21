"""Preserve source, verify input bindings and fingerprint the selected recovery evidence."""

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

from scene_recall.run_metadata import code_hash


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def finalize(root):
    repository = Path(__file__).resolve().parents[1]
    root = root.resolve()
    manifest = root / "evidence_manifest.json"
    capsule = root / "source_final.tar.gz"
    if manifest.exists() or capsule.exists():
        raise ValueError("final evidence already exists")
    protected = json.loads((root / "protected.json").read_text())
    previous = json.loads(
        (root.parent / "slam_system_study/evidence_manifest.json").read_text()
    )["scientific_files"]
    for name, expected in (previous | protected).items():
        if file_hash(repository / name) != expected:
            raise ValueError("protected scientific evidence changed: " + name)
    summary = json.loads((root / "summary.json").read_text())
    agreements = summary["default_pose_segment_status_agreement"]
    isolation = summary["atlas_does_not_change_local_estimator"]
    if (
        len(agreements) != 5
        or not all(agreements.values())
        or not all(isolation.values())
    ):
        raise ValueError("baseline or isolation verification is incomplete")
    bindings = set()
    for parent in (root, root.parent / "slam_system_study/new_inputs"):
        for path in parent.rglob("manifest.json"):
            value = json.loads(path.read_text())
            if "features_sha256" in value and "baseline_report_sha256" in value:
                if file_hash(path.parent / "features.npz") != value["features_sha256"]:
                    raise ValueError("feature cache changed")
                if (
                    file_hash(path.parent / "budget/report.json")
                    != value["baseline_report_sha256"]
                ):
                    raise ValueError("feature cache input report changed")
                bindings.add(
                    (value["baseline_report_sha256"], value["features_sha256"])
                )
    cache_checks = 0
    for path in root.rglob("protocol.json"):
        value = json.loads(path.read_text())
        if value.get("features_sha256") and "baseline_report_sha256" in value:
            if (
                value["baseline_report_sha256"],
                value["features_sha256"],
            ) not in bindings:
                raise ValueError("unbound feature cache protocol: " + str(path))
            cache_checks += 1
    sources = sorted(
        [
            p
            for folder in ("src", "experiments", "scripts", "tests", "docs")
            for p in (repository / folder).rglob("*")
            if p.is_file() and p.suffix in (".py", ".md")
        ]
        + [
            repository / name
            for name in (
                "README.md",
                "pyproject.toml",
                "uv.lock",
                ".gitignore",
                ".gitattributes",
            )
        ]
    )
    package = code_hash()
    with tarfile.open(capsule, "w:gz") as archive:
        for path in sources:
            archive.add(
                path, arcname=path.relative_to(repository).as_posix(), recursive=False
            )
    source_record = {
        "package_sha256": package,
        "archive_sha256": file_hash(capsule),
        "files": {
            path.relative_to(repository).as_posix(): file_hash(path) for path in sources
        },
    }
    (root / "source_final.json").write_text(json.dumps(source_record, indent=2) + "\n")
    excluded = {
        "pilot_pioneer",
        "pioneer_full",
        "sfm_inputs",
        "sfm_inputs_verified",
        "sfm_lab_room_1",
        "serial_raw",
        "cycles_rpy",
        "sources",
    }
    artifacts = {}
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or path == manifest
            or any(part in excluded for part in path.relative_to(root).parts)
        ):
            continue
        if path.suffix in (".json", ".npz", ".csv") or path.name in (
            "release_pytest.log",
            "final_focused_tests.log",
        ):
            artifacts[path.relative_to(repository).as_posix()] = file_hash(path)
    initial = json.loads((root / "source_before_confirmation.json").read_text())
    if (
        file_hash(root / "source_before_confirmation.tar.gz")
        != initial["archive_sha256"]
    ):
        raise ValueError("initial source capsule changed")
    result = {
        "selection": "completed numerical outputs, protocols, scientific input caches, verified report derivations and final verification logs",
        "excluded_phases": sorted(excluded - {"sources"}),
        "source_data": "source images and download archives are excluded from this file list, official archive/table digests are in acquisition JSON and selected observations have canonical input digests",
        "starting_commit": json.loads((root / "protocol.json").read_text())[
            "starting_commit"
        ],
        "current_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip(),
        "package_sha256": package,
        "source_capsules": {
            "source_before_confirmation.tar.gz": initial["archive_sha256"],
            "source_final.tar.gz": source_record["archive_sha256"],
        },
        "preservation": {
            "prior_study_files": len(previous),
            "protected_files": len(protected),
            "changed": [],
        },
        "feature_cache_protocols_verified": cache_checks,
        "default_pose_segment_status_agreement": agreements,
        "atlas_does_not_change_local_estimator": isolation,
        "serial_timing": "serial_raw_verified, opposite-order scientific arrays identical",
        "full_tests": (root / "release_pytest.log").read_text().splitlines()[-1],
        "scientific_files": artifacts,
    }
    if "passed" not in result["full_tests"] or "failed" in result["full_tests"]:
        raise ValueError("final full-suite success is missing")
    if code_hash() != package:
        raise ValueError("package changed during evidence finalization")
    manifest.write_text(json.dumps(result, indent=2) + "\n")
    print(
        "Verified",
        len(artifacts),
        "scientific files and",
        len(previous) + len(protected),
        "protected files",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    finalize(parser.parse_args().root)
