"""Verify preserved baselines, exact local contracts and repeat scientific arrays."""

import argparse
import json
from pathlib import Path

import numpy as np

from scene_recall.run_metadata import code_hash, sha256


def compare_npz(first, second):
    count = 0
    with (
        np.load(first, allow_pickle=False) as a,
        np.load(second, allow_pickle=False) as b,
    ):
        if a.files != b.files:
            raise ValueError(f"archive fields changed: {first}")
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key], err_msg=f"{first}: {key}")
            count += 1
    return count


def verify(root):
    protected = json.loads((root / "protected_baselines.json").read_text())
    for name, digest in protected.items():
        if sha256(Path(name)) != digest:
            raise ValueError(f"protected scientific baseline changed: {name}")
    cases = json.loads((root / "cases.json").read_text())
    contracts = {}
    for name, specification in cases.items():
        source = Path(specification["local_run"])
        target = root / "final" / name
        protocol = json.loads((target / "protocol.json").read_text())
        if protocol["package_sha256"] != code_hash():
            raise ValueError(f"package identity changed since final replay: {name}")
        if protocol["driver_sha256"] != sha256(
            Path(__file__).with_name("slam_loop_graph.py")
        ):
            raise ValueError(f"driver identity changed since final replay: {name}")
        original = json.loads((source / "report.json").read_text())
        for policy in protocol["policies"]:
            derived = json.loads((target / policy / "report.json").read_text())
            if any(
                derived[key] != original[key]
                for key in ("frames", "evaluation", "config", "source")
            ):
                raise ValueError(f"local contract changed: {name}/{policy}")
            if sha256(target / policy / "trajectory.npz") != sha256(
                source / "trajectory.npz"
            ):
                raise ValueError(f"local trajectory bytes changed: {name}/{policy}")
        with (
            np.load(source / "atlas.npz") as a,
            np.load(target / "forest_mean" / "atlas.npz") as b,
        ):
            for key in ("T_component_depth", "component_id"):
                np.testing.assert_array_equal(a[key], b[key])
        contracts[name] = {
            "unchanged_local_policies": len(protocol["policies"]),
            "exact_original_forest": True,
        }
    repeat_arrays = {}
    original = root / "final" / "outages" / "outage_xyz80"
    for repeated in ("final_repeat_xyz80", "final_repeat_raw_xyz80"):
        destination = root / repeated
        protocol = json.loads((destination / "protocol.json").read_text())
        count = sum(
            compare_npz(
                original / policy / "atlas.npz", destination / policy / "atlas.npz"
            )
            for policy in protocol["policies"]
        )
        count += sum(
            compare_npz(original / name, destination / name)
            for name in (
                "constraints_mean.npz",
                "constraints_votes.npz",
                "accepted_registrations.npz",
            )
        )
        for name in ("events_mean.json", "events_votes.json"):
            if json.loads((original / name).read_text()) != json.loads(
                (destination / name).read_text()
            ):
                raise ValueError(f"repeat event diagnostics changed: {repeated}/{name}")
        repeat_arrays[repeated] = count
    map_arrays = sum(
        compare_npz(
            root / "maps" / "natural" / "xyz" / name, root / "map_repeat_xyz" / name
        )
        for name in ("predictions.npz", "meshes.npz")
    )
    first = json.loads((root / "maps/natural/xyz/report.json").read_text())
    second = json.loads((root / "map_repeat_xyz/report.json").read_text())
    if first["scores"] != second["scores"] or first["maps"] != second["maps"]:
        raise ValueError("repeat reconstruction scores or cohorts changed")
    result = {
        "protected_files_unchanged": len(protected),
        "case_contracts": contracts,
        "replay_repeat_arrays": repeat_arrays,
        "reconstruction_repeat_arrays": map_arrays,
        "package_sha256": code_hash(),
        "driver_sha256": sha256(Path(__file__)),
    }
    (root / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    args = parser.parse_args()
    verify(args.study_root)
