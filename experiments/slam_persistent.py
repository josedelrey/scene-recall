"""Select graph seeds using accepted frontend observation history.

This experiment changes seed admission only. Historical measurements influence
selection but never enter the joint objective a second time. Existing graph
identities keep priority. Image-cell round robin preserves spatial distribution.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import incremental_slam
import numpy as np

from scene_recall.multiview.temporal import bind_depth
from scene_recall.run_metadata import code_hash, sha256
from scene_recall.slam.system import IncrementalSLAM, SLAMConfig


def persistent_seeds(pixels, candidates, lengths, budget, shape):
    """Prefer longer accepted histories within each image cell, without future views."""
    bins = {}
    for i in candidates:
        cell = tuple(np.floor(pixels[i] / [shape[1], shape[0]] * [8, 6]).astype(int))
        bins.setdefault(cell, []).append(int(i))
    for values in bins.values():
        values.sort(key=lambda i: (-lengths[i], i))
    return [
        values[j]
        for j in range(max(map(len, bins.values()), default=0))
        for _, values in sorted(bins.items())
        if j < len(values)
    ][:budget]


class PersistentSeedsSLAM(IncrementalSLAM):
    """Serial experimental admission adapter preserving raw feature identities."""

    def _ingest(self, observation, calibration, depth_pose, features, motion):
        samples = bind_depth(
            features.pixels, observation.depth, calibration, motion, self.config.timing
        )
        key = observation.frame_index
        owners = self.frontend.owners[key]
        candidates = [
            i
            for i in np.flatnonzero(owners >= 0)
            if int(owners[i]) not in self.identities and samples.source_index[i] >= 0
        ]
        lengths = {
            i: len(self.frontend.landmarks[int(owners[i])].observations)
            for i in candidates
        }
        selected = persistent_seeds(
            features.pixels,
            candidates,
            lengths,
            max(0, self.config.max_landmarks - len(self.points)),
            calibration.rgb_shape,
        )
        admitted = owners.copy()
        admitted[np.setdiff1d(candidates, selected).astype(int)] = -1
        # Limit this serial boundary to selected fresh seeds. Restore the actual
        # frontend ownership immediately, preserving its next-frame associations.
        with (
            patch.dict(self.frontend.owners, {key: admitted}),
            patch("scene_recall.slam.system.bind_depth", return_value=samples),
        ):
            result = super()._ingest(
                observation, calibration, depth_pose, features, motion
            )
        result["new_seed_history_lengths"] = [lengths[i] for i in selected]
        return result


def run(baseline, output, names, features=None):
    if output.exists():
        raise ValueError("choose a fresh output directory")
    policies = {
        "persistent": SLAMConfig(),
        "persistent_turnover": SLAMConfig(marginalization="sqrt", stale_keyframes=2),
    }
    identity = sha256(Path(__file__))
    output.parent.mkdir(parents=True, exist_ok=True)
    admission_protocol = output.with_name(output.name + "_admission_protocol.json")
    if admission_protocol.exists():
        raise ValueError("admission protocol already exists")
    admission_protocol.write_text(
        json.dumps(
            {
                "phase": "frozen before evaluation",
                "wrapper_sha256": identity,
                "selection": "Descending accepted frontend observation count within 8x6 cells, spatial round robin, existing graph identities first",
                "policies": {name: asdict(policies[name]) for name in names},
            },
            indent=2,
        )
        + "\n"
    )
    # run_case creates output and records the common replay protocol itself.
    with (
        patch.object(incremental_slam, "variants", return_value=policies),
        patch.object(incremental_slam, "IncrementalSLAM", PersistentSeedsSLAM),
    ):
        result = incremental_slam.run_case(baseline, output, names, features=features)
    if sha256(Path(__file__)) != identity:
        raise ValueError("persistent selection changed during evaluation")
    (output / "persistent_protocol.json").write_text(
        json.dumps(
            {
                "wrapper_sha256": identity,
                "implementation": code_hash(),
                "policies": {name: asdict(policies[name]) for name in names},
                "selection": "Descending accepted frontend observation count within 8x6 cells, spatial round robin, existing graph identities first",
                "historical_measurements": "Used for seed admission only. Newly admitted raw factors belong to the current frame.",
            },
            indent=2,
        )
        + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--features", type=Path)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("persistent", "persistent_turnover"),
        default=["persistent", "persistent_turnover"],
    )
    args = parser.parse_args()
    run(args.baseline_run, args.output_dir, args.variants, args.features)


if __name__ == "__main__":
    main()
