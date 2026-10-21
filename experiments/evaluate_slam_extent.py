"""Post hoc full-extent reconstruction under a fixed, enlarged submap budget.

All existing optimizer inputs, training cadence, targets and fusion settings
stay fixed. Missing training references withhold the predetermined whole chunk.
Do not pool this extent diagnostic with the primary bounded-submap populations.
"""

import argparse
import json
from pathlib import Path

from refinement_tsdf import evaluate_tsdf

from scene_recall.mapping.heldout import run_heldout
from scene_recall.mapping.voxel import MappingConfig
from scene_recall.run_metadata import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=["control", "native", "wide"])
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("choose a fresh output directory")
    args.output_dir.mkdir(parents=True)
    protocol = {
        "phase": "frozen before extent scoring",
        "budget": 128,
        "variants": args.variants,
        "driver_sha256": sha256(Path(__file__)),
        "type": "post hoc extent diagnostic, without changing optimizer inputs or replacing missing-reference chunks",
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n"
    )
    result = {}
    for name in args.variants:
        paths = {"budget": args.baseline_run, name: args.case_root / name}
        fusion = run_heldout(
            list(paths.values()),
            args.output_dir / ("maps_" + name),
            config=MappingConfig(pixel_stride=8, max_keyframes_per_submap=128),
        )
        result[name] = evaluate_tsdf(
            paths, list(paths), args.output_dir / ("tsdf_" + name), fusion
        )
        print(name, result[name]["status"], flush=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
