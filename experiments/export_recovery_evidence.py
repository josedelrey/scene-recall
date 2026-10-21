"""Copy verified study exports with explicit separation of inherited input headers.

Original experiment outputs are preserved byte for byte. This derivation changes
only report headers. Poses, failures, per-frame diagnostics and all measurements
are copied unchanged and checked by digest. Input-run timing and measurement
summaries remain labeled context rather than measurements of the new replay.
"""

import argparse
import json
import shutil
from pathlib import Path

from slam_recovery import execution_header

from scene_recall.run_metadata import sha256


def export(root, output):
    if output.exists():
        raise ValueError("choose a fresh export directory")
    comparisons = []
    for path in sorted(root.rglob("comparison.json")):
        if path.relative_to(root).parts[0].startswith(("serial", "verified_exports")):
            continue
        if "pilot" in str(path) or path.parent.name in (
            "pioneer_full",
            "sfm_lab_room_1",
        ):
            continue
        value = json.loads(path.read_text())
        if "policies" in value:
            comparisons.append((path, value))
    output.mkdir(parents=True)
    records = {}
    for path, comparison in comparisons:
        relative = path.parent.relative_to(root)
        target = output / relative
        target.mkdir(parents=True)
        protocol = json.loads((path.parent / "protocol.json").read_text())
        for name in ("protocol.json", "comparison.json"):
            shutil.copyfile(path.parent / name, target / name)
        for policy, result in comparison["policies"].items():
            source = path.parent / policy
            destination = target / policy
            destination.mkdir()
            report = json.loads((source / "report.json").read_text())
            current = execution_header(
                report, protocol, result["process_and_atlas_seconds"]
            )
            current["report_derivation"] = {
                "type": "label inherited input-run header context",
                "original_report_sha256": sha256(source / "report.json"),
                "export_driver_sha256": sha256(Path(__file__)),
            }
            unchanged = {}
            for item in source.iterdir():
                if item.is_file() and item.name != "report.json":
                    shutil.copyfile(item, destination / item.name)
                    expected = sha256(item)
                    assert sha256(destination / item.name) == expected
                    unchanged[item.name] = expected
            (destination / "report.json").write_text(
                json.dumps(current, indent=2, allow_nan=False) + "\n"
            )
            assert (
                current["frames"] == report["frames"]
                and current["evaluation"] == report["evaluation"]
            )
            records[str(relative / policy)] = {
                "original_report_sha256": sha256(source / "report.json"),
                "derived_report_sha256": sha256(destination / "report.json"),
                "unchanged_files": unchanged,
            }
    (output / "derivation.json").write_text(
        json.dumps(
            {
                "scope": "report headers only, original evidence retained",
                "runs": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export(args.root, args.output_dir)
